# ✅ 설치 필요: pip install kiwipiepy openai python-dotenv pandas
import os
import re
import json
import time
import pandas as pd
from fractions import Fraction
from dotenv import load_dotenv
from openai import OpenAI
from kiwipiepy import Kiwi  # ✅ MeCab 대신 Kiwi 사용

# ===== 설정 =====
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
tagger = Kiwi()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "recipes.csv")

SEASONING_KEYWORDS = ["간장", "된장", "고추장", "설탕", "소금", "식초", "후추", "맛술", "참기름", "들기름", "식용유",
                      "다진마늘", "다진대파", "고춧가루", "올리고당", "레몬즙", "마요네즈", "머스터드", "케찹",
                      "파슬리", "카레가루", "와사비", "미림", "멸치액젓", "매실청", "매실액", "조청", "청주",
                      "국간장", "연두부", "버터"]
UNIT_CANDIDATES = ["g", "ml", "큰술", "작은술", "숟가락", "스푼", "Ts", "T", "개", "알", "쪽", "줄기", "장", "통",
                   "조각", "대", "줌", "모", "봉지", "컵", "소량", "약간", "적당량", "마리"]
TEXTUAL_UNITS = ["소량", "약간", "적당량"]
UNNECESSARY_PHRASES = ["기호에 따라", "선택사항", "취향껏", "원하는 만큼"]
IGNORE_ITEMS = ["고명", "양념", "양념장"]

# ===== 재료 파싱 =====
def is_seasoning(item):
    return any(k in item for k in SEASONING_KEYWORDS)

def clean_item_name(item):
    item = re.sub(r"\([^)]*\)", "", item)
    item = re.sub(r"\s+", " ", item)
    for phrase in UNNECESSARY_PHRASES:
        item = item.replace(phrase, "")
    item = re.sub(r"(개당|씩)", "", item)

    tokens = tagger.tokenize(item.strip())
    filtered = [t.form for t in tokens if t.tag.startswith("N") and len(t.form) > 1 and t.form not in UNIT_CANDIDATES]
    return " ".join(filtered).strip()

def extract_amount_and_unit_from_parenthesis(text):
    match = re.search(r"\(([\d/\.]+)\s*([가-힣a-zA-Z]+)?\)", text)
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

def extract_ingredient_info_light(raw):
    if not isinstance(raw, str):
        return {"재료": [], "조미료": []}
    raw = raw.replace("●", "").replace("·", ",").replace(":", " : ").replace("\n", ",")
    parts = re.split(r"[,\n·•]", raw)
    parts = [p.strip() for p in parts if p.strip()]
    재료, 조미료 = [], []
    for part in parts:
        parsed = parse_single_ingredient_brute(part)
        if parsed["item"]:
            if is_seasoning(parsed["item"]):
                조미료.append(parsed)
            else:
                재료.append(parsed)
    return {"재료": 재료, "조미료": 조미료}

# ===== GPT 요청 =====
def ask_gpt_for_single_recipe(ingredients: list) -> dict:
    재료_문장 = ", ".join(ingredients)
    prompt = f"""
다음 재료들이 모두 포함된 간단한 요리를 하나 추천해줘: {재료_문장}

- 아래와 같은 JSON 구조로만 응답해줘:
{{
  "레시피이름": {{
    "재료": [{{"item": ..., "amount": ..., "unit": ...}}, ...],
    "조미료": [{{"item": ..., "amount": ..., "unit": ...}}, ...]
  }}
}}

- 설명 없이 JSON만 응답해.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 한국 요리를 잘 아는 요리 도우미야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            timeout=10
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print("[GPT ERROR]", e)
        return {}

# ===== 추천 메인 =====
def is_match(보유: str, 레시피_아이템: str) -> bool:
    return 보유.strip() == 레시피_아이템.strip()

def run_recipe_recommendation(보유_재료: list):
    df = pd.read_csv(CSV_PATH).dropna(subset=["재료"])
    if "재료_JSON" not in df.columns:
        df["재료_JSON"] = None

    recipe_dict = {}
    updated_indices = []

    for idx, row in df.iterrows():
        parsed_light = extract_ingredient_info_light(row["재료"])
        레시피_재료 = [x["item"].strip() for x in parsed_light["재료"] if x["item"]]
        is_In = all(any(is_match(보유, r_item) for r_item in 레시피_재료) for 보유 in 보유_재료)
        if is_In:
            if pd.notnull(row["재료_JSON"]):
                parsed = json.loads(row["재료_JSON"])
            else:
                parsed = extract_ingredient_info_light(row["재료"])
                df.at[idx, "재료_JSON"] = json.dumps(parsed, ensure_ascii=False)
                updated_indices.append(idx)
            recipe_dict[row["요리이름"]] = parsed

    # GPT 보완
    if len(recipe_dict) < 5:
        print("[INFO] GPT 보완 시작")
        start_time = time.time()
        while len(recipe_dict) < 5 and time.time() - start_time < 15:
            gpt_recipe = ask_gpt_for_single_recipe(보유_재료)
            for name, data in gpt_recipe.items():
                if name not in recipe_dict:
                    recipe_dict[name] = data
                    print(f"[GPT] '{name}' 추가됨")
                    재료_텍스트 = ", ".join(f'{i["item"]} {i["amount"]}{i["unit"]}' for i in data["재료"] + data["조미료"])
                    df = pd.concat([df, pd.DataFrame([{"요리이름": name, "재료": 재료_텍스트, "재료_JSON": json.dumps(data, ensure_ascii=False)}])], ignore_index=True)
                if len(recipe_dict) >= 5:
                    break

    if updated_indices or len(recipe_dict) > 0:
        df.to_csv(CSV_PATH, index=False)
        print(f"[INFO] recipes.csv 저장 완료 (총 {len(df)}개)")

    print(f"[DEBUG] 추천 레시피 수: {len(recipe_dict)}")
    return recipe_dict

# ===== 테스트 =====
if __name__ == "__main__":
    보유 = ["감자", "양파"]
    결과 = run_recipe_recommendation(보유)
    print(json.dumps(결과, ensure_ascii=False, indent=2))
