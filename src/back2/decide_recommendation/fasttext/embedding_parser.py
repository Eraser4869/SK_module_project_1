import re
from konlpy.tag import Okt
from fractions import Fraction

okt = Okt()

SEASONING_KEYWORDS = [
    "간장", "된장", "고추장", "설탕", "소금", "식초", "후추", "맛술", "참기름",
    "들기름", "식용유", "다진마늘", "다진대파", "고춧가루", "올리고당", "레몬즙",
    "마요네즈", "머스터드", "케찹", "파슬리", "카레가루", "와사비", "미림",
    "멸치액젓", "매실청", "매실액", "조청", "청주", "국간장", "연두부", "버터"
]

UNIT_CANDIDATES = [
    "g", "ml", "큰술", "작은술", "컵", "ts", "T", "개", "쪽", "장", "줄기", "통", "알",
    "조금", "조각", "대", "줄", "덩어리", "줌", "근", "모", "봉지", "소량"
]

def is_seasoning(item: str) -> bool:
    return any(k in item for k in SEASONING_KEYWORDS)

def clean_item_name(item: str) -> str:
    item = re.sub(r"\(.*?\)", "", item)
    item = item.replace("  ", " ").strip()
    tokens = okt.nouns(item)
    return " ".join([t for t in tokens if len(t) > 1 and t != "찜"])

def parse_single_ingredient_brute(text: str):
    for unit in UNIT_CANDIDATES:
        pattern = rf"(.+?)\s*([\d\/\.]+)?\s*{unit}"
        match = re.match(pattern, text)
        if match:
            item = clean_item_name(match.group(1))
            raw_amt = match.group(2)
            try:
                amount = float(Fraction(raw_amt)) if raw_amt else None
            except:
                amount = None
            return {
                "item": item,
                "amount": amount,
                "unit": unit
            }

    return {
        "item": clean_item_name(text),
        "amount": None,
        "unit": None
    }


def extract_ingredient_info(raw: str):
    raw = raw.replace("●", "").replace("·", "").replace(":", " : ").replace("\n", " ")
    raw = raw.replace("재료 :", "").replace("양념장 :", "").replace("소스 :", "").replace("양념 :", "")
    parts = re.split(r"[,\n·•]", raw)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 0 and len(parts[0].split()) > 3:
        parts = parts[1:]

    재료, 조미료 = [], []

    for part in parts:
        parsed = parse_single_ingredient_brute(part)
        if parsed["item"]:
            if is_seasoning(parsed["item"]):
                조미료.append(parsed)
            else:
                재료.append(parsed)

    return {"재료": 재료, "조미료": 조미료}
