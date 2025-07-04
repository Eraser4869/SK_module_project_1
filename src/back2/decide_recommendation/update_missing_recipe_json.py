import os
import pandas as pd
import json
from back2.decide_recommendation.api.ingredient_utils import extract_ingredient_info

def update_missing_json():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "./api/recipes.csv")

    df = pd.read_csv(csv_path)

    # 조건: 재료 존재 + 재료_JSON이 NaN이거나 비어있는 문자열
    mask = df["재료"].notna() & (df["재료_JSON"].isna() | (df["재료_JSON"].str.strip() == ""))
    targets = df[mask]
    print(f"[INFO] 업데이트 대상 레시피 수: {len(targets)}")

    for idx, row in targets.iterrows():
        print(f"[INFO] '{row['요리이름']}' 파싱 중...")
        try:
            parsed = extract_ingredient_info(row["재료"])
            df.at[idx, "재료_JSON"] = json.dumps(parsed, ensure_ascii=False)
        except Exception as e:
            print(f"[ERROR] '{row['요리이름']}' 파싱 실패:", e)

    df.to_csv(csv_path, index=False)
    print("[INFO] recipes.csv 저장 완료 ✅")

if __name__ == "__main__":
    update_missing_json()
