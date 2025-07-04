import json
from gensim.models import FastText

# 1. 재료 목록 로드 (nutrition_items.json)
with open("nutrition_items.json", "r", encoding="utf-8") as f:
    ingredient_list = json.load(f)

# 2. 중복 제거 및 리스트 반복 구성 (학습 데이터 부족 보완용)
ingredients = list(set(ingredient_list))
training_sentences = [ingredients] * 10

# 3. FastText 모델 학습
model = FastText(
    sentences=training_sentences,
    vector_size=100,
    window=3,
    min_count=1,
    sg=1,
    epochs=100
)

# 4. 모델 저장
model.save("ingredient_fasttext.model")

print("✅ 모델 학습 및 저장 완료")

