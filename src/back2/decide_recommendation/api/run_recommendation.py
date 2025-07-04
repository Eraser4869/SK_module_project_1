import os
import time
import pandas as pd
import json
from openai import OpenAI
from dotenv import load_dotenv
from src.back2.decide_recommendation.api.ingredient_utils import extract_ingredient_info, extract_ingredient_info_light

# ✅ 최소 추천 레시피 개수 설정
MIN_RECIPE_COUNT = 5
TIME_LIMIT_SECONDS = 15

# ✅ GPT API 설정
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ✅ 재료 일치 판단
def is_match(보유: str, 레시피_아이템: str) -> bool:
    return 보유.strip() == 레시피_아이템.strip()

# ✅ GPT에 1개 레시피 요청
def ask_gpt_for_single_recipe(보유재료: list) -> dict:
    재료_문장 = ", ".join(보유재료)
    prompt = f"""
다음 재료들이 모두 포함된 간단한 요리를 하나 추천해줘: {재료_문장}

- 아래와 같은 JSON 구조로만 응답해줘:
{{
  "레시피이름": {{
    "재료": [{{"item": ..., "amount": ..., "unit": ...}}, ...],
    "조미료": [{{"item": ..., "amount": ..., "unit": ...}}, ...]
  }}
}}

- key는 반드시 실제 요리 이름이어야 해 (예: "감자계란덮밥")
- 단위와 수치는 현실적으로 구성해줘.
- 설명 없이 오직 JSON만 반환해.
- 재료는 같은 재료가 번복 되면 안됨.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 한국 요리를 잘 아는 요리 도우미야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            timeout=10  # 응답 지연 방지
        )
        json_str = response.choices[0].message.content.strip()
        return json.loads(json_str)
    except Exception as e:
        print("[GPT ERROR]", e)
        return {}

# ✅ 메인 추천 함수
def run_recipe_recommendation(보유_재료: list):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "recipes.csv")

    df = pd.read_csv(csv_path).dropna(subset=["재료"])
    if "재료_JSON" not in df.columns:
        df["재료_JSON"] = None

    recipe_dict = {}
    updated_indices = []

    for idx, row in df.iterrows():
        parsed_light = extract_ingredient_info_light(row["재료"])
        레시피_재료 = [x["item"].strip() for x in parsed_light["재료"] if x["item"]]

        is_In = all(
            any(is_match(보유.strip(), r_item) for r_item in 레시피_재료)
            for 보유 in 보유_재료
        )
        if is_In:
            if pd.notnull(row["재료_JSON"]):
                parsed = json.loads(row["재료_JSON"])
            else:
                parsed = extract_ingredient_info(row["재료"])
                df.at[idx, "재료_JSON"] = json.dumps(parsed, ensure_ascii=False)
                updated_indices.append(idx)

            recipe_dict[row["요리이름"]] = {
                "재료": parsed["재료"],
                "조미료": parsed["조미료"]
            }

    # ✅ GPT 보완 로직 (15초 제한, 1개씩 누적 요청)
    if len(recipe_dict) < MIN_RECIPE_COUNT:
        print("[INFO] 추천 레시피가 부족하여 GPT로 보완 생성 시작")
        start_time = time.time()

        while len(recipe_dict) < MIN_RECIPE_COUNT:
            if time.time() - start_time > TIME_LIMIT_SECONDS:
                print("[INFO] GPT 보완 시간 초과. 확보된 레시피까지만 사용")
                break

            gpt_recipe = ask_gpt_for_single_recipe(보유_재료)

            if not isinstance(gpt_recipe, dict):
                print("[ERROR] GPT 응답이 dict가 아님")
                break

            for name, data in gpt_recipe.items():
                if name in recipe_dict:
                    continue
                if isinstance(data, dict) and "재료" in data and "조미료" in data:
                    recipe_dict[name] = data
                    print(f"[INFO] GPT 레시피 '{name}' 추가됨")

                    try:
                        재료_텍스트 = ", ".join(
                            [f'{i["item"]} {i["amount"]}{i["unit"]}' for i in data["재료"] + data["조미료"]]
                        )
                        재료_JSON = json.dumps(data, ensure_ascii=False)
                        new_row = pd.DataFrame({
                            "요리이름": [name],
                            "재료": [재료_텍스트],
                            "재료_JSON": [재료_JSON]
                        })
                        df = pd.concat([df, new_row], ignore_index=True)
                    except Exception as e:
                        print(f"[ERROR] GPT 레시피 저장 실패: {e}")

                if len(recipe_dict) >= MIN_RECIPE_COUNT:
                    break

    # ✅ CSV 저장
    if updated_indices or len(recipe_dict) > 0:
        df.to_csv(csv_path, index=False)
        print(f"[INFO] recipes.csv 저장 완료 (총 {len(df)}개 레시피)")

    print(f"[DEBUG] 총 추천 레시피 수: {len(recipe_dict)}")
    return recipe_dict

# ✅ 테스트 실행
if __name__ == "__main__":
    test_in = ["감자", "계란"]
    result = run_recipe_recommendation(test_in)
    print(json.dumps(result, ensure_ascii=False, indent=2))
