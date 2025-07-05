import os
import re
import json
import pandas as pd
from fractions import Fraction
from dotenv import load_dotenv
from openai import OpenAI
from konlpy.tag import Okt  # ✅ 변경된 부분

# ===== 환경 설정 =====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
tagger = Okt()  # ✅ 변경된 부분
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "recipes.csv")

# ===== 기준 키워드 =====
SEASONING_KEYWORDS = ["간장", "된장", "고추장", "설탕", "소금", "식초", "후추", "맛술", "참기름", "들기름", "식용유",
                      "다진마늘", "다진대파", "고춧가루", "올리고당", "레몬즙", "마요네즈", "머스터드", "케찹",
                      "파슬리", "카레가루", "와사비", "미림", "멸치액젓", "매실청", "매실액", "조청", "청주",
                      "국간장", "연두부", "버터"]
UNIT_CANDIDATES = ["g", "ml", "큰술", "작은술", "숟가락", "스푼", "Ts", "T", "개", "알", "쪽", "줄기", "장", "통",
                   "조각", "대", "줌", "모", "봉지", "컵", "소량", "약간", "적당량", "마리"]
TEXTUAL_UNITS = ["소량", "약간", "적당량"]
UNNECESSARY_PHRASES = ["기호에 따라", "선택사항", "취향껏", "원하는 만큼"]
IGNORE_ITEMS = ["고명", "순두부 사과 소스", "양념", "양념장", "스트로베리 샐러드", "황태해장국", "방울토마토 소박이"]

# ===== 전처리 함수들 =====
def is_seasoning(item):
    return any(k in item for k in SEASONING_KEYWORDS)

def clean_item_name(item):
    item = re.sub(r"\([^)]*\)", "", item)
    item = re.sub(r"\s+", " ", item)
    for phrase in UNNECESSARY_PHRASES:
        item = item.replace(phrase, "")
    item = re.sub(r"(개당|씩)", "", item)
    try:
        tokens = tagger.nouns(item.strip())
    except:
        return item.strip()
    filtered = [t for t in tokens if len(t) > 1 and t != "찜"]
    while filtered and filtered[0] in UNIT_CANDIDATES:
        filtered = filtered[1:]
    return " ".join(filtered).strip()

def extract_amount_and_unit_from_parenthesis(text):
    match = re.search(r"\(([⅐-⅞\d/\.]+)\s*([가-힣a-zA-Z]+)?\)", text)
    if match:
        amt, unit = match.group(1), match.group(2)
        try:
            return float(Fraction(amt)), unit
        except:
            return None, unit
    return None, None

def parse_single_ingredient_brute(text):
    bracket_amount, bracket_unit = extract_amount_and_unit_from_parenthesis(text)
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
            amount = amount or bracket_amount
            unit = unit or bracket_unit
            return {"item": item, "amount": amount, "unit": unit}
    for unit in TEXTUAL_UNITS:
        if unit in text:
            return {"item": clean_item_name(text.replace(unit, "")), "amount": 1, "unit": unit}
    return {"item": clean_item_name(text), "amount": bracket_amount, "unit": bracket_unit}

def expand_parenthetical_items(text):
    match = re.match(r"(.+?)\(([^()]+)\)", text)
    if match:
        base, inner = match.group(1).strip(), match.group(2)
        parts = [x.strip() for x in inner.split(",")]
        return [f"{x} {base}" for x in parts]
    return [text]

def extract_with_gpt(ingredient_lines):
    prompt = "다음 재료들을 각각 item, amount, unit 으로 구조화해서 JSON 배열 형식으로 반환해줘. 단위와 수량이 없으면 null로:\n"
    for line in ingredient_lines:
        prompt += f"- {line.strip()}\n"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 한국 요리 재료를 구조화하는 도우미야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print("[ERROR] GPT API 오류:", e)
        return None

def extract_ingredient_info(raw: str) -> dict:
    if not isinstance(raw, str):
        return {"재료": [], "조미료": []}

    print("[INFO] 원문 전처리 시작")
    raw = raw.replace("●", "").replace("·", ",").replace(":", " : ").replace("\n", ",")
    raw = raw.replace("재료 :", "").replace("양념장 :", "").replace("소스 :", "").replace("양념 :", "")
    parts = re.split(r"[,\n·•]", raw)
    parts = [p.strip() for p in parts if p.strip()]
    print(f"[INFO] 재료 항목 수: {len(parts)}")

    if len(parts) > 0 and (len(parts[0].split()) > 3 or not any(u in parts[0] for u in UNIT_CANDIDATES)):
        print(f"[INFO] 첫 항목 제거: {parts[0]}")
        parts = parts[1:]

    print("[INFO] 파싱 시작")
    재료, 조미료 = [], []
    for part in parts:
        expanded = expand_parenthetical_items(part)
        for p in expanded:
            parsed = parse_single_ingredient_brute(p)
            if parsed["item"]:
                if is_seasoning(parsed["item"]):
                    조미료.append(parsed)
                else:
                    재료.append(parsed)
    print(f"[INFO] 파싱 완료 - 재료 {len(재료)}개, 조미료 {len(조미료)}개")

    결측 = [x for x in (재료 + 조미료) if (x["amount"] is None or x["unit"] is None) and x["item"] not in IGNORE_ITEMS]

    if 결측:
        print(f"[INFO] GPT 보정 시작 - 결측값 {len(결측)}개:")
        for i, x in enumerate(결측, 1):
            print(f"    {i}. item: {x['item']}, amount: {x['amount']}, unit: {x['unit']}")
        gpt_fixed = extract_with_gpt([raw])
        try:
            gpt_result = json.loads(gpt_fixed)
            print(f"[INFO] GPT 보정 완료 - 총 {len(gpt_result)}개")
            return {
                "재료": [x for x in gpt_result if not is_seasoning(x["item"])],
                "조미료": [x for x in gpt_result if is_seasoning(x["item"])]
            }
        except Exception as e:
            print("[ERROR] GPT 보정 실패:", e)
    else:
        print("[INFO] GPT 보정 생략 - 무시 가능한 부재료들")

    return {"재료": 재료, "조미료": 조미료}

def extract_ingredient_info_light(raw: str) -> dict:
    if not isinstance(raw, str):
        return {"재료": [], "조미료": []}

    raw = raw.replace("●", "").replace("·", ",").replace(":", " : ").replace("\n", ",")
    raw = raw.replace("재료 :", "").replace("양념장 :", "").replace("소스 :", "").replace("양념 :", "")
    parts = re.split(r"[,\n·•]", raw)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 0 and (len(parts[0].split()) > 3 or not any(u in parts[0] for u in UNIT_CANDIDATES)):
        parts = parts[1:]

    재료, 조미료 = [], []
    for part in parts:
        expanded = expand_parenthetical_items(part)
        for p in expanded:
            parsed = parse_single_ingredient_brute(p)
            if parsed["item"]:
                if is_seasoning(parsed["item"]):
                    조미료.append(parsed)
                else:
                    재료.append(parsed)

    return {"재료": 재료, "조미료": 조미료}

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    targets = df[df["재료"].notna()]
    print("[INFO] 전처리 대상 레시피 수:", len(targets))

    fixed_data = []
    for idx, row in enumerate(targets.itertuples(), 1):
        parsed = extract_ingredient_info(row.재료)
        fixed_data.append(json.dumps(parsed, ensure_ascii=False))

    df.loc[targets.index, "재료_JSON"] = fixed_data
    df.to_csv(CSV_PATH, index=False)
    print(f"[INFO] 모든 전처리 완료 및 저장 ✅ (총 {len(targets)}개)")
    print(f"[INFO] CSV 저장 경로: {CSV_PATH}")
