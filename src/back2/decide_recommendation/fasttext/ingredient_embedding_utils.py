from gensim.models import FastText
from konlpy.tag import Okt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

okt = Okt()

# ▶ 현재 파일 기준 상대경로로 모델 경로 설정
MODEL_PATH = os.path.join(os.path.dirname(__file__), "ingredient_fasttext.model")
model = FastText.load(MODEL_PATH)

def parse_ingredients(text):
    """형태소 분석기로 사용자 입력 재료 파싱"""
    tokens = okt.nouns(text)
    return [token for token in tokens if token in model.wv]

def get_vector_from_text(text):
    """주어진 문장(재료 문자열)에서 평균 벡터 추출"""
    tokens = parse_ingredients(text)
    if not tokens:
        return None
    vectors = [model.wv[token] for token in tokens]
    return np.mean(vectors, axis=0)

def build_recipe_vectors(recipe_dict):
    """전체 레시피 딕셔너리에서 평균 임베딩 벡터를 구함"""
    recipe_vectors = {}
    for name, items in recipe_dict.items():
        ingredients = [x['item'] for x in items['재료']] + [x['item'] for x in items['조미료']]
        vec = get_vector_from_text(" ".join(ingredients))
        if vec is not None:
            recipe_vectors[name] = vec
    return recipe_vectors

def recommend_recipes(user_input, recipe_vectors, top_n=5):
    """사용자 입력과 레시피 벡터 간 유사도를 기반으로 추천"""
    user_vec = get_vector_from_text(user_input)
    if user_vec is None:
        return []

    similarities = {
        name: cosine_similarity([user_vec], [vec])[0][0]
        for name, vec in recipe_vectors.items()
    }

    return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
