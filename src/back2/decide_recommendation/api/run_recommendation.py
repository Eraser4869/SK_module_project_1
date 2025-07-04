import os
import pandas as pd
import json
from simple_classifier_module import SimpleRecipeClassifier
from ingredient_utils import extract_ingredient_info, extract_ingredient_info_light

def is_match(보유: str, 레시피_아이템: str) -> bool:
    return 보유.strip() == 레시피_아이템.strip()

def run_recipe_recommendation(보유_재료: list):
    # 1. 현재 파일 기준 recipes.csv 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "recipes.csv")

    # 2. CSV 로드
    df = pd.read_csv(csv_path).dropna(subset=["재료"])

    # 3. 재료_JSON 컬럼이 없다면 생성
    if "재료_JSON" not in df.columns:
        df["재료_JSON"] = None

    # 4. 추천 대상 레시피 필터링 + 보유 재료 비교
    recipe_dict = {}
    updated_indices = []

    for idx, row in df.iterrows():
        parsed_light = extract_ingredient_info_light(row["재료"])
        레시피_재료 = [x["item"].strip() for x in parsed_light["재료"] if x["item"]]

        is_In = all(
            any(is_match(보유.strip(), r_item) for r_item in 레시피_재료)
            for 보유 in 보유_재료
        )

        print(f"[DEBUG] 요리명: {row['요리이름']}")
        print(f"[DEBUG] 레시피 재료: {레시피_재료}")
        print(f"[DEBUG] 보유 재료: {보유_재료}")
        print(f"[DEBUG] 포함 여부: {is_In}")
        print("----")

        if is_In:
            # 4-1. 재료_JSON이 존재하면 사용
            if pd.notnull(row["재료_JSON"]):
                parsed = json.loads(row["재료_JSON"])
            else:
                # 4-2. 없으면 GPT로 파싱 후 저장
                parsed = extract_ingredient_info(row["재료"])
                df.at[idx, "재료_JSON"] = json.dumps(parsed, ensure_ascii=False)
                updated_indices.append(idx)

            recipe_dict[row["요리이름"]] = {
                "재료": parsed["재료"],
                "조미료": parsed["조미료"]
            }

    # 5. GPT 수행된 친구들만 CSV 저장
    if updated_indices:
        df.to_csv(csv_path, index=False)
        print(f"[INFO] {len(updated_indices)}건 재료_JSON 생성 및 저장 완료 ✅")

    print(f"[DEBUG] 총 추천 대상 레시피 수: {len(recipe_dict)}")

    # 6. 분류기 실행
    classifier = SimpleRecipeClassifier(recipe_dict)
    return classifier.recommend(보유_재료)

# ✅ 테스트용 메인
if __name__ == "__main__":
    try:
        test_in = ["감자", "계란"]
        result = run_recipe_recommendation(test_in)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[예외 발생]: {e}")
