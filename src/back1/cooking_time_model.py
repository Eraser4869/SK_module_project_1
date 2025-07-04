import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import re
import json




# 데이터 불러오기 및 전처리 
df = pd.read_csv('RECIPE_SEARCH.csv')

# 조리시간 파싱 함수
def convert_time_to_minutes(time_str):
    if pd.isna(time_str):
        return np.nan
    time_str = str(time_str)
    hours = re.findall(r'(\d+)\s*시간', time_str)
    minutes = re.findall(r'(\d+)\s*분', time_str)
    total = 0
    if hours:
        total += int(hours[0]) * 60
    if minutes:
        total += int(minutes[0])
    return total if total > 0 else np.nan

df['COOKING_TIME_MIN'] = df['CKG_TIME_NM'].apply(convert_time_to_minutes)

# 필요한 컬럼 필터링 및 결측 제거
df_model = df[['CKG_KND_ACTO_NM', 'CKG_DODF_NM', 'COOKING_TIME_MIN']].dropna()
df_model = pd.get_dummies(df_model, columns=['CKG_KND_ACTO_NM', 'CKG_DODF_NM'])

X = df_model.drop('COOKING_TIME_MIN', axis=1)
y = df_model['COOKING_TIME_MIN']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 학습 및 저장
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"[모델 학습 완료] 평균 절대 오차: {mae:.2f} 분")

joblib.dump(model, 'cooking_time_model.pkl')
joblib.dump(X.columns.tolist(), 'feature_columns.pkl')
print("[모델 및 feature 저장 완료]")

# 레시피명으로 kind / level 추출
def get_kind_level_by_recipe(recipe_name: str, df_raw: pd.DataFrame):
    match = df_raw[df_raw['CKG_NM'].str.strip() == recipe_name.strip()]
    if not match.empty:
        row = match.iloc[0]
        return row['CKG_KND_ACTO_NM'], row['CKG_DODF_NM']
    else:
        print(f"[경고] 레시피명 '{recipe_name}'을 찾을 수 없습니다.")
        return None, None

# 예측 함수 
def predict_with_model(kind: str, level: str) -> float:
    model = joblib.load('cooking_time_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')

    sample = {col: 0 for col in feature_columns}
    kind_col = f'CKG_KND_ACTO_NM_{kind}'
    level_col = f'CKG_DODF_NM_{level}'

    if kind_col in sample:
        sample[kind_col] = 1
    else:
        print(f"[경고] '{kind}'에 해당하는 feature가 없습니다.")

    if level_col in sample:
        sample[level_col] = 1
    else:
        print(f"[경고] '{level}'에 해당하는 feature가 없습니다.")

    sample_df = pd.DataFrame([sample])
    return model.predict(sample_df)[0]

# JSON 파싱 + 예측 + 사용자 선호 반영
def parse_json_and_predict(json_str, preference_dict):
    data = json.loads(json_str)
    results = []

    # preference_dict 예시
    # {
    #   "희망_식단": ["다이어트", "채식"],
    #   "희망_조리시간": ["간단", "보통"],
    #   "희망_난이도": ["쉬움"]
    # }


    for recipe_name in data:
        kind, level = get_kind_level_by_recipe(recipe_name, df)
        if not kind or not level:
            continue
        predicted_time = predict_with_model(kind, level)
        ingredients = [i.get("item", "") for i in data[recipe_name].get("재료", [])]

        # 선호도
        print(f"사용자 선호: {preference_dict}")

        results.append({
            "recipe_name": recipe_name,
            "kind": kind,
            "level": level,
            "predicted_cooking_time": round(predicted_time, 2),
            "ingredients": ingredients
        })

    return results


if __name__ == "__main__":
    # 테스트용 JSON
    json_str = '''
    {
        "소고기떡국": {
            "재료": [
                {"item": "떡국떡", "amount": 400, "unit": "g"},
                {"item": "다진소고기", "amount": 100, "unit": "g"},
                {"item": "대파", "amount": 1, "unit": "대"}
            ]
        },
        "된장수육": {
            "재료": [
                {"item": "삼겹살", "amount": 500, "unit": "g"},
                {"item": "된장", "amount": 2, "unit": "큰술"}
            ]
        }
    }
    '''

    # 테스트용 사용자 선호도 dict
    preference_dict = {
        "희망_식단": ["다이어트", "저염"],
        "희망_조리시간": ["간단"],
        "희망_난이도": ["쉬움"]
    }

    results = parse_json_and_predict(json_str, preference_dict)
    for r in results:
        print(f"[{r['recipe_name']}] ({r['kind']}, {r['level']}) → 예측 시간: {r['predicted_cooking_time']}분")
