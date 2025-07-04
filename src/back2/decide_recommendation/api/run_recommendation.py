import os
import pandas as pd
import json
from simple_classifier_module import SimpleRecipeClassifier
from ingredient_utils import extract_ingredient_info  # ✅ 전처리 함수는 따로 import

def run_recipe_recommendation(보유_재료: list):
    # 현재 파일 기준 상대경로로 CSV 로드
    csv_path = os.path.join(os.path.dirname(__file__), "recipes.csv")
    df = pd.read_csv(csv_path).dropna(subset=["재료"])

    # 전체 레시피 파싱
    recipe_dict = {}
    for _, row in df.iterrows():
        parsed = extract_ingredient_info(row["재료"])
        recipe_dict[row["요리이름"]] = {
            "재료": parsed["재료"],
            "조미료": parsed["조미료"]
        }

    # 분류기 실행
    classifier = SimpleRecipeClassifier(recipe_dict)
    return classifier.recommend(보유_재료)

# ✅ 메인 테스트 실행
if __name__ == "__main__":
    test_재료 = ["감자", "양파", "달걀"]
    결과 = run_recipe_recommendation(test_재료)
    print(json.dumps(결과, ensure_ascii=False, indent=2))
