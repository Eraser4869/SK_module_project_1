# AI 기반 재료 매칭을 사용하는 레시피 분류기
# sentence-transformers를 사용한 의미적 유사도 매칭 시스템
# 
# 주요 기능:
# 1. AI 기반 재료명 매칭 (한국어 특화)
# 2. 영양소 계산 및 분석
# 3. 다양한 식단 분류 (다이어트, 저탄고지, 저염, 채식)
# 4. 단위 변환 및 정량화

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher  # 문자열 유사도 계산용 (기본 매칭에서 사용)

# AI 라이브러리 import (설치 필요: pip install sentence-transformers torch)
# try-except 구문을 사용하여 라이브러리 설치 여부 확인
try:
    from sentence_transformers import SentenceTransformer, util  # 문장 임베딩 모델
    import torch  # PyTorch 딥러닝 프레임워크
    AI_AVAILABLE = True  # AI 사용 플래그
except ImportError:
    # 라이브러리가 설치되지 않은 경우 기본 매칭으로 폴백
    AI_AVAILABLE = False
    print("AI 라이브러리가 설치되지 않았습니다. 기본 매칭만 사용합니다.")
    


class SmartRecipeClassifier:
    # AI 기반 지능형 레시피 분류 클래스
    # 재료명을 자동으로 매칭하고 영양소를 계산하여 다양한 식단으로 분류

    def __init__(self, nutrition_csv_path: str, use_ai: bool = True):
        # 클래스 초기화 메서드
        # nutrition_csv_path: 데이터 파일 경로
        # use_ai: AI 매칭 사용 여부 (default: True)
        
        # AI 사용 여부 결정 (라이브러리 설치 상태 & 사용자 선택)
        self.use_ai = use_ai and AI_AVAILABLE
        
        # private 함수 초기화
        self.nutrition_db = self._load_nutrition_data(nutrition_csv_path)  # 영양소 DB 로드
        self.fallback_nutrition = self._create_fallback_nutrition()        # 기본 영양소 데이터
        self.unit_conversion = self._create_unit_conversion()              # 단위 변환 테이블
        self.classification_rules = self._define_classification_rules()    # 분류 규칙
        
        # AI 모델 초기화 (사용 가능한 경우에만)
        if self.use_ai:
            self._initialize_ai_matcher()
        else: #이거야?
            print("기본 매칭 모드로 실행됩니다.")
    


    def _load_nutrition_data(self, csv_path: str) -> pd.DataFrame:
        # 국가표준식품성분표 CSV 파일을 로드하는 메서드
        # 기본 재료 중심의 상세한 영양소 정보를 포함
        # csv_path: CSV 파일의 경로
        # 반환값: 재료명(item)을 인덱스로 하는 DataFrame
        
        # CSV 파일 읽기
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # 'item' 컬럼이 비어있는 행 제거 (이중 확인)
        # 국가표준식품성분표는 'item' 컬럼에 재료명이 저장됨
        df = df.dropna(subset=['item'])
        
        # 'item' 컬럼을 인덱스로 설정 (빠른 검색을 위해)
        df.set_index('item', inplace=True)
        
        return df
    


    def _initialize_ai_matcher(self):
        # AI 매칭 시스템을 초기화하는 메서드
        # 한국어 특화 문장 임베딩 모델을 로드하고 식품 데이터를 벡터화
        
        print("AI 매칭 시스템 초기화 중...")
        
        # 한국어 특화 임베딩 모델 로드
        # jhgan/ko-sbert-nli: Hugging face에서 가져온 문장 임베딩 학습 모델. 한국어는 이게 제일 잘되는듯...
        model_name = 'jhgan/ko-sbert-nli'
        self.embedding_model = SentenceTransformer(model_name)
        
        # 캐시 파일 경로 설정 (임베딩 재계산 방지)
        self.embeddings_cache_path = 'nutrition_embeddings.pt'  # PyTorch 텐서 저장용
        self.items_cache_path = 'nutrition_items.json'          # 식품 목록 저장용
        

        # 식품명 리스트 생성 (DB + 기본 재료)
        # list() 함수로 pandas Index를 일반 리스트로 변환
        self.food_items = list(self.nutrition_db.index) + list(self.fallback_nutrition.keys())  #이거야?
        
        # 임베딩 로드 또는 새로 생성
        self._load_or_create_embeddings()
        print("AI 매칭 시스템 준비 완료!")
    


    def _load_or_create_embeddings(self):
        # 임베딩 캐시를 로드하거나 새로 생성하는 메서드
        # 한번 생성 후에는 캐시된 임베딩 사용
        
        # 캐시 파일이 존재하고 식품 목록이 동일한지 확인
        if (os.path.exists(self.embeddings_cache_path) and 
            os.path.exists(self.items_cache_path)):
            
            # 기존 식품 목록과 현재 목록 비교
            with open(self.items_cache_path, 'r', encoding='utf-8') as f:
                cached_items = json.load(f)  # JSON 파일에서 리스트 로드
            
            # 목록이 동일하면 캐시된 임베딩 사용
            if cached_items == self.food_items:
                # torch.load(): PyTorch 텐서 파일 로드
                self.food_embeddings = torch.load(self.embeddings_cache_path)
                print("캐시된 AI 임베딩을 로드했습니다.")
                return
        
        # 새로운 임베딩 생성 (최초 실행시)
        print("AI 임베딩 생성 중... (최초 실행시 시간이 걸립니다)")
        
        # 문장 임베딩 생성 (의미적 벡터 표현으로 변환)
        # encode() 메서드: 텍스트를 고차원 벡터로 변환
        self.food_embeddings = self.embedding_model.encode(
            self.food_items,            # 인코딩할 텍스트 리스트
            convert_to_tensor=True,     # PyTorch 텐서로 변환
            show_progress_bar=True      # 진행률 표시
        )
        
        # 생성된 임베딩을 캐시에 저장
        torch.save(self.food_embeddings, self.embeddings_cache_path)
        
        # 식품 목록도 JSON 파일로 저장
        with open(self.items_cache_path, 'w', encoding='utf-8') as f:
            json.dump(self.food_items, f, ensure_ascii=False, indent=2)
        
        print("AI 임베딩을 캐시에 저장했습니다.")
    


# 이거야!
    def _create_fallback_nutrition(self) -> Dict[str, Dict]:
        # 기본 영양소 데이터를 정의하는 비상용 메서드
        # AI가 매칭하지 못하거나 DB에 없는 재료들을 위한 백업 데이터
        # 국가표준식품성분표의 컬럼명에 맞춰 구성
        # 재료명을 키로 하고 영양소 정보를 값으로 하는 딕셔너리
        
        return {
            # 기본 조미료들 (100g 당 영양소)
            # 국가표준식품성분표 컬럼명 사용: 당류, 총 식이섬유 등
            '소금': {
                'kcal': 0,                    # 칼로리
                'carb_g': 0,                  # 탄수화물(g)
                'fat_g': 0,                   # 지방(g)
                'protein_g': 0,               # 단백질(g)
                'sodium_mg': 40000,           # 나트륨(mg)
                '당류': 0,                    # 당류(g) - 데이터 파일의 컬럼명
                '총  식이섬유': 0             # 총 식이섬유(g) - 공백 포함 주의
            },
            '설탕': {
                'kcal': 387, 'carb_g': 99.8, 'fat_g': 0, 'protein_g': 0, 
                'sodium_mg': 1, '당류': 99.8, '총  식이섬유': 0
            },
            '물': {
                'kcal': 0, 'carb_g': 0, 'fat_g': 0, 'protein_g': 0, 
                'sodium_mg': 0, '당류': 0, '총  식이섬유': 0
            },
            '간장': {
                'kcal': 50, 'carb_g': 5, 'fat_g': 0, 'protein_g': 8, 
                'sodium_mg': 5000, '당류': 5, '총  식이섬유': 0
            },
            '참기름': {
                'kcal': 884, 'carb_g': 0, 'fat_g': 100, 'protein_g': 0, 
                'sodium_mg': 0, '당류': 0, '총  식이섬유': 0
            },
            '올리브오일': {
                'kcal': 884, 'carb_g': 0, 'fat_g': 100, 'protein_g': 0, 
                'sodium_mg': 2, '당류': 0, '총  식이섬유': 0
            },
            '고춧가루': {
                'kcal': 282, 'carb_g': 56, 'fat_g': 12, 'protein_g': 12, 
                'sodium_mg': 35, '당류': 28, '총  식이섬유': 35
            },
            
            # 기본 재료들 (AI가 매칭하지 못할 때 대비용)
            '닭고기': {
                'kcal': 165, 'carb_g': 0, 'fat_g': 3.6, 'protein_g': 31, 
                'sodium_mg': 74, '당류': 0, '총  식이섬유': 0
            },
            '돼지고기': {
                'kcal': 250, 'carb_g': 0, 'fat_g': 14, 'protein_g': 27, 
                'sodium_mg': 65, '당류': 0, '총  식이섬유': 0
            },
            '쇠고기': {
                'kcal': 200, 'carb_g': 0, 'fat_g': 8, 'protein_g': 26, 
                'sodium_mg': 55, '당류': 0, '총  식이섬유': 0
            },
            '양파': {
                'kcal': 40, 'carb_g': 9, 'fat_g': 0.1, 'protein_g': 1.1, 
                'sodium_mg': 4, '당류': 4.2, '총  식이섬유': 1.7
            },
            '마늘': {
                'kcal': 149, 'carb_g': 33, 'fat_g': 0.5, 'protein_g': 6.4, 
                'sodium_mg': 17, '당류': 1, '총  식이섬유': 2.1
            },
            '토마토': {
                'kcal': 18, 'carb_g': 3.9, 'fat_g': 0.2, 'protein_g': 0.9, 
                'sodium_mg': 5, '당류': 2.6, '총  식이섬유': 1.2
            },
        }
    


    def _create_unit_conversion(self) -> Dict:
        # 단위 변환 테이블을 생성하는 메서드
        # 이미 정제되어 오지만, 이중 확인
        # 다양한 단위를 그램(g)으로 통일하기 위한 변환 정보
        # 반환값: 변환 정보를 담은 딕셔너리
        
        return {
            # 기본 단위 변환 비율 (그램 기준)
            'conversion_rates': {
                'g': 1.0,       # 그램 (기준 단위)
                'kg': 1000.0,   # 킬로그램 = 1000g
                'mg': 0.001,    # 밀리그램 = 0.001g
                'ml': 1.0,      # 밀리리터 (물 기준, 밀도 1.0)
                'cc': 1.0,      # 시시 (ml와 동일)
                'l': 1000.0,    # 리터 = 1000ml
                '큰술': 15.0,   # 1큰술 = 15ml
                '작은술': 5.0,  # 1작은술 = 5ml
                '컵': 200.0,    # 1컵 = 200ml (한국 기준)
                '국자': 50.0    # 1국자 = 50ml (평균)
            },
            
            # 개수 단위를 그램으로 변환 (재료별 평균 무게)
            'piece_weights': {
                '계란': {'개': 50, '알': 50},    # 계란 1개 = 50g
                '마늘': {'쪽': 3, '개': 20},     # 마늘 1쪽 = 3g, 1개 = 20g
                '양파': {'개': 200},             # 양파 1개 = 200g
                '토마토': {'개': 150},           # 토마토 1개 = 150g
                '감자': {'개': 150},             # 감자 1개 = 150g
                '두부': {'모': 300},             # 두부 1모 = 300g
                '김': {'장': 3}                 # 김 1장 = 3g
            },
            
            # 재료별 밀도 (부피를 무게로 변환시 사용)
            'densities': {
                '물': 1.0,          # 물의 밀도 (기준)
                '간장': 1.15,       # 간장은 물보다 약간 무거움
                '참기름': 0.92,     # 기름류는 물보다 가벼움
                '올리브오일': 0.92,
                '밀가루': 0.6,      # 가루류는 공기를 많이 포함
                '설탕': 0.8,
                '소금': 1.2,        # 소금은 물보다 무거움
                '고춧가루': 0.3     # 고춧가루는 매우 가벼움
            }
        }
    


    def _define_classification_rules(self) -> Dict:
        # 레시피 분류 규칙을 정의하는 메서드
        # 각 식단 유형별로 100g당 영양소 기준을 설정
        # 반환값: 분류 규칙을 담은 딕셔너리
        
        return {
            # 다이어트 식단 기준 (저칼로리, 저지방, 고단백, 저당)
            '다이어트': {
                'kcal_per_100g_max': 250,       # 최대 250kcal/100g
                'fat_per_100g_max': 10,         # 최대 10g 지방/100g
                'protein_per_100g_min': 8,      # 최소 8g 단백질/100g
                'sugar_per_100g_max': 15        # 최대 15g 당류/100g
            },
            
            # 저탄고지(케토) 식단 기준 (저탄수화물, 고지방, 적정단백질)
            '저탄고지': {
                'carb_per_100g_max': 15,        # 최대 15g 탄수화물/100g
                'fat_per_100g_min': 10,         # 최소 10g 지방/100g
                'protein_per_100g_min': 10,     # 최소 10g 단백질/100g
                'sugar_per_100g_max': 8         # 최대 8g 당류/100g
            },
            
            # 저염 식단 기준
            '저염': {
                'sodium_per_100g_max': 400      # 최대 400mg 나트륨/100g
            },
            
            # 채식 식단 기준 (동물성 재료 제외)
            '채식': {
                'exclude_keywords': [
                    # 육류 관련 키워드들
                    '돼지', '소', '닭', '오리', '양', 
                    # 해산물 관련 키워드들
                    '생선', '연어', '참치', '새우', '게', '조개', '굴', 
                    '문어', '오징어', '멸치',
                    # 가공육 관련 키워드들
                    '삼겹살', '갈비', '치킨', '햄', '소시지', '베이컨'
                ]
            }
        }
    


    def find_best_match_ai(self, ingredient_name: str, threshold: float = 0.6) -> Tuple[str, float, Dict]:
        # AI 기반 재료 매칭을 수행하는 메서드
        # 입력 재료명과 가장 유사한 DB 내 재료를 찾아 반환
        # ingredient_name: 찾을 재료명
        # threshold: 유사도 임계값 (기본 0.6)
        # 반환값: (매칭된 재료명, 유사도, 영양소 정보) 튜플
        
        # 입력값 정제 (앞뒤 공백 제거)
        ingredient_name = ingredient_name.strip()
        
        # 1단계: 정확한 매칭 확인 (완전히 동일한 이름)
        if ingredient_name in self.food_items:
            return self._get_nutrition_data(ingredient_name, 1.0)
        
        # 2단계: AI 의미적 유사도 계산
        # 입력 재료명을 임베딩 벡터로 변환
        query_embedding = self.embedding_model.encode(ingredient_name, convert_to_tensor=True)
        
        # 코사인 유사도 계산 (의미적 유사성 측정)
        # util.cos_sim(): 두 벡터 간의 코사인 유사도 계산 (범위: -1 ~ 1)
        # 높을수록 의미적으로 유사함
        similarities = util.cos_sim(query_embedding, self.food_embeddings)[0]
        
        # 가장 유사한 항목 찾기
        # torch.argmax(): 최대값의 인덱스를 반환
        best_idx = torch.argmax(similarities).item()      # .item()으로 스칼라 값 추출
        best_similarity = similarities[best_idx].item()   # 해당 인덱스의 유사도 값
        
        # 임계값 확인 (설정된 기준 이상인 경우만 매칭 성공)
        if best_similarity >= threshold:
            best_match = self.food_items[best_idx]
            print(f"AI 매칭: '{ingredient_name}' → '{best_match}' (유사도: {best_similarity:.3f})")
            return self._get_nutrition_data(best_match, best_similarity)
        
        # 3단계: 매칭 실패시 기본값 반환 (물로 대체)
        print(f"매칭 실패: '{ingredient_name}' (유사도: {best_similarity:.3f})")
        return '물', best_similarity, self.fallback_nutrition['물']


#이거야
    def find_best_match_basic(self, ingredient_name: str) -> Tuple[str, float, Dict]:
        # 기본 문자열 매칭을 수행하는 메서드 (AI 미사용시)
        # 단순한 문자열 포함 관계와 유사도로 매칭
        # ingredient_name: 찾을 재료명
        # 반환값: (매칭된 재료명, 유사도, 영양소 정보) 튜플
        
        ingredient_name = ingredient_name.strip()
        
        # 1단계: 정확한 매칭 확인
        if ingredient_name in self.nutrition_db.index:
            nutrition = self.nutrition_db.loc[ingredient_name].to_dict()  # Series를 dict로 변환
            return ingredient_name, 1.0, nutrition
        
        # 2단계: 부분 매칭 (문자열 포함 관계)
        best_match = None
        best_score = 0
        
        # DB의 모든 식품명과 비교
        for db_name in self.nutrition_db.index:
            # 상호 포함 관계 확인 (양방향)
            if ingredient_name in db_name or db_name in ingredient_name:
                # SequenceMatcher: 두 문자열 간의 유사도 계산 (0~1)
                score = SequenceMatcher(None, ingredient_name.lower(), db_name.lower()).ratio()
                if score > best_score:
                    best_score = score
                    best_match = db_name
        
        # 유사도가 0.6 이상인 경우 매칭 성공
        if best_match and best_score > 0.6:
            nutrition = self.nutrition_db.loc[best_match].to_dict()
            return best_match, best_score, nutrition
        
        # 3단계: Fallback 데이터에서 검색
        for fallback_name, nutrition in self.fallback_nutrition.items():
            if ingredient_name in fallback_name or fallback_name in ingredient_name:
                return fallback_name, 0.5, nutrition
        
        # 4단계: 최종 기본값 (물)
        return '물', 0.1, self.fallback_nutrition['물']
    


    def _get_nutrition_data(self, item_name: str, similarity: float) -> Tuple[str, float, Dict]:
        # 특정 재료의 영양소 데이터를 가져오는 메서드
        # item_name: 재료명
        # similarity: 매칭 유사도
        # 반환값: (재료명, 유사도, 영양소 정보) 튜플
        
        # 1순위: 메인 DB에서 확인
        if item_name in self.nutrition_db.index:
            # pandas Series를 dict로 변환
            nutrition = self.nutrition_db.loc[item_name].to_dict()
            return item_name, similarity, nutrition
        

        # 이거야
        # 2순위: Fallback 데이터에서 확인
        if item_name in self.fallback_nutrition:
            return item_name, similarity, self.fallback_nutrition[item_name]
        
        # 3순위: 기본값 (물)
        return '물', 0.1, self.fallback_nutrition['물']
    


    def find_best_match(self, ingredient_name: str) -> Tuple[str, float, Dict]:
        # 재료 매칭의 메인 인터페이스 메서드
        # AI 사용 여부에 따라 적절한 매칭 방식 선택
        # ingredient_name: 찾을 재료명
        # 반환값: (매칭된 재료명, 유사도, 영양소 정보) 튜플
        
        if self.use_ai:
            # AI 매칭 사용
            return self.find_best_match_ai(ingredient_name)
        else:
            # 이거야
            # 기본 문자열 매칭 사용
            return self.find_best_match_basic(ingredient_name)
    

    '''지우려면 위의 _create_unit_conversion 메서드 포함 사용된거 다 '''
    def convert_to_grams(self, amount: float, unit: str, ingredient_name: str = '') -> float:
        # 다양한 단위를 그램으로 변환하는 메서드
        # amount: 양 (숫자)
        # unit: 단위 (문자열)
        # ingredient_name: 재료명 (단위 변환시 참고용)
        # 반환값: 그램 단위로 변환된 양
        
        # 입력값 정제 (소문자 변환, 공백 제거)
        unit = unit.lower().strip()
        ingredient_name = ingredient_name.lower().strip()
        
        conversion = self.unit_conversion
        
        # 1단계: 기본 무게/부피 단위 변환
        if unit in conversion['conversion_rates']:
            # 기본 변환 적용
            base_amount = amount * conversion['conversion_rates'][unit]
            
            # 부피 단위의 경우 밀도 적용 (ml → g 변환)
            if unit in ['ml', 'cc', 'l', '큰술', '작은술', '컵']:
                # 재료별 밀도 확인
                for ingredient, density in conversion['densities'].items():
                    if ingredient in ingredient_name:
                        return base_amount * density  # 부피 × 밀도 = 무게
                return base_amount  # 기본 밀도 1.0 (물 기준)
            
            return base_amount
        
        # 2단계: 개수 단위 변환 (개, 알, 쪽 등)
        for ingredient, weights in conversion['piece_weights'].items():
            # 재료명이 포함되고 해당 단위가 있는 경우
            if ingredient in ingredient_name and unit in weights:
                return amount * weights[unit]  # 개수 × 개당 무게
        
        # 3단계: 기본 개수 단위 (평균값 사용)
        default_weights = {
            '개': 100,    # 일반적인 개체 1개 평균 100g
            '알': 50,     # 알 종류 1개 평균 50g
            '쪽': 3,      # 쪽 단위 평균 3g
            '장': 3,      # 장 단위 평균 3g
            '모': 300     # 모 단위 평균 300g
        }
        if unit in default_weights:
            return amount * default_weights[unit]
        
        # 4단계: 변환 불가능한 경우 원래 값 반환
        return amount
    


    def calculate_nutrition(self, recipe_data: Dict) -> Dict:
        # 레시피의 전체 영양소를 계산하는 메서드
        # 모든 재료의 영양소를 합산하여 총 영양소와 100g당 영양소 계산
        # 국가표준식품성분표의 컬럼명에 맞춰 처리
        # recipe_data: 레시피 정보가 담긴 딕셔너리
        # 반환값: 영양소 계산 결과를 담은 딕셔너리
        
        # 총 영양소 초기화 (모든 영양소를 0으로 설정)
        # 국가표준식품성분표 컬럼명 사용
        total_nutrition = {
            'kcal': 0,              # 총 칼로리
            'carb_g': 0,            # 총 탄수화물(g)
            'fat_g': 0,             # 총 지방(g)
            'protein_g': 0,         # 총 단백질(g)
            'sodium_mg': 0,         # 총 나트륨(mg)
            '당류': 0,              # 총 당류(g) - 국가표준 컬럼명
            '총  식이섬유': 0       # 총 식이섬유(g) - 공백 포함 주의
        }
        
        total_weight_g = 0      # 총 중량(g)
        matching_details = []   # 매칭 상세 정보 저장용 리스트
        
        # 재료와 조미료를 모두 합쳐서 처리
        # get() 메서드: 키가 없으면 빈 리스트 반환
        all_ingredients = recipe_data.get('재료', []) + recipe_data.get('조미료', [])
        
        # 각 재료별로 영양소 계산
        for ingredient in all_ingredients:
            # 재료 정보 추출 (딕셔너리에서 값 가져오기)
            item_name = ingredient.get('item', '')      # 재료명
            amount = ingredient.get('amount', 0)        # 양
            unit = ingredient.get('unit', 'g')          # 단위
            
            # 유효하지 않은 재료는 건너뛰기
            if not item_name or amount <= 0:
                continue
            
            # AI 또는 기본 매칭으로 재료 찾기
            matched_name, similarity, nutrition = self.find_best_match(item_name)
            
            # 단위를 그램으로 변환
            weight_g = self.convert_to_grams(amount, unit, item_name)
            total_weight_g += weight_g
            
            # 100g당 영양소에서 실제 사용량에 맞는 영양소 계산
            # multiplier: 100g 대비 실제 사용량 비율
            multiplier = weight_g / 100
            
            # 각 영양소별로 계산하여 총 영양소에 누적
            # 국가표준식품성분표의 컬럼명 매핑 처리
            for nutrient in total_nutrition.keys():
                # 영양소가 데이터에 있고 null이 아닌 경우
                if nutrient in nutrition and nutrition[nutrient] is not None:
                    # pandas의 NaN 값 체크 (결측값 처리)
                    if not pd.isna(nutrition[nutrient]):
                        total_nutrition[nutrient] += nutrition[nutrient] * multiplier
            
            # 매칭 상세 정보 저장
            matching_details.append({
                'original': item_name,      # 원본 재료명
                'matched': matched_name,    # 매칭된 재료명
                'similarity': similarity,   # 매칭 유사도
                'weight_g': weight_g        # 그램 단위 중량
            })
        
        # 100g당 영양소 계산
        nutrition_per_100g = {}
        if total_weight_g > 0:  # 0으로 나누기 방지
            for nutrient, total_value in total_nutrition.items():
                # (총 영양소 / 총 중량) × 100 = 100g당 영양소
                nutrition_per_100g[nutrient] = (total_value / total_weight_g) * 100
        
        # 결과를 딕셔너리로 반환
        return {
            'total_nutrition': total_nutrition,         # 총 영양소
            'nutrition_per_100g': nutrition_per_100g,   # 100g당 영양소
            'total_weight_g': total_weight_g,           # 총 중량
            'matching_details': matching_details        # 매칭 상세 정보
        }
    


    def classify_recipe(self, recipe_name: str, recipe_data: Dict) -> Dict:
        # 레시피를 다양한 식단 카테고리로 분류하는 메서드
        # recipe_name: 레시피 이름
        # recipe_data: 레시피 데이터 (재료, 조미료 정보 포함)
        # 반환값: 분류 결과를 담은 딕셔너리
        
        print(f"\n'{recipe_name}' 분석 중...")
        
        # 영양소 계산 수행
        nutrition_result = self.calculate_nutrition(recipe_data)
        nutrition_100g = nutrition_result['nutrition_per_100g']
        
        # 분류 결과와 이유를 저장할 딕셔너리
        classifications = {}  # 각 카테고리별 분류 결과 (True/False)
        reasons = {}         # 각 카테고리별 분류 이유
        
        # === 다이어트 식단 분류 ===
        diet_rules = self.classification_rules['다이어트']
        
        # 각 조건별 확인 (모든 조건을 만족해야 다이어트 식단)
        # 국가표준식품성분표 컬럼명에 맞춰 조정
        diet_checks = [
            nutrition_100g.get('kcal', 0) <= diet_rules['kcal_per_100g_max'],      # 칼로리 체크
            nutrition_100g.get('fat_g', 0) <= diet_rules['fat_per_100g_max'],      # 지방 체크
            nutrition_100g.get('protein_g', 0) >= diet_rules['protein_per_100g_min'], # 단백질 체크
            nutrition_100g.get('당류', 0) <= diet_rules['sugar_per_100g_max']        # 당류 체크 (국가표준 컬럼명)
        ]
        
        # all() 함수: 모든 조건이 True일 때만 True 반환
        classifications['다이어트'] = all(diet_checks)
        
        # 실패한 조건들을 찾아서 이유 생성
        failed_conditions = []
        if not diet_checks[0]: 
            failed_conditions.append(f"칼로리 초과({nutrition_100g.get('kcal', 0):.1f}kcal)")
        if not diet_checks[1]: 
            failed_conditions.append(f"지방 초과({nutrition_100g.get('fat_g', 0):.1f}g)")
        if not diet_checks[2]: 
            failed_conditions.append(f"단백질 부족({nutrition_100g.get('protein_g', 0):.1f}g)")
        if not diet_checks[3]: 
            failed_conditions.append(f"당류 초과({nutrition_100g.get('당류', 0):.1f}g)")
        
        # join() 메서드: 리스트의 요소들을 문자열로 연결
        reasons['다이어트'] = ", ".join(failed_conditions) if failed_conditions else "모든 조건 만족"
        

        # === 저탄고지(케토) 식단 분류 ===
        keto_rules = self.classification_rules['저탄고지']
        keto_checks = [
            nutrition_100g.get('carb_g', 0) <= keto_rules['carb_per_100g_max'],    # 탄수화물 체크
            nutrition_100g.get('fat_g', 0) >= keto_rules['fat_per_100g_min'],      # 지방 체크
            nutrition_100g.get('protein_g', 0) >= keto_rules['protein_per_100g_min'], # 단백질 체크
            nutrition_100g.get('당류', 0) <= keto_rules['sugar_per_100g_max']        # 당류 체크 (국가표준 컬럼명)
        ]
        classifications['저탄고지'] = all(keto_checks)
        
        failed_conditions = []
        if not keto_checks[0]: 
            failed_conditions.append(f"탄수화물 초과({nutrition_100g.get('carb_g', 0):.1f}g)")
        if not keto_checks[1]: 
            failed_conditions.append(f"지방 부족({nutrition_100g.get('fat_g', 0):.1f}g)")
        if not keto_checks[2]: 
            failed_conditions.append(f"단백질 부족({nutrition_100g.get('protein_g', 0):.1f}g)")
        if not keto_checks[3]: 
            failed_conditions.append(f"당류 초과({nutrition_100g.get('당류', 0):.1f}g)")
        reasons['저탄고지'] = ", ".join(failed_conditions) if failed_conditions else "모든 조건 만족"
        classifications['저탄고지'] = all(keto_checks)
        
        failed_conditions = []
        if not keto_checks[0]: 
            failed_conditions.append(f"탄수화물 초과({nutrition_100g.get('carb_g', 0):.1f}g)")
        if not keto_checks[1]: 
            failed_conditions.append(f"지방 부족({nutrition_100g.get('fat_g', 0):.1f}g)")
        if not keto_checks[2]: 
            failed_conditions.append(f"단백질 부족({nutrition_100g.get('protein_g', 0):.1f}g)")
        if not keto_checks[3]: 
            failed_conditions.append(f"당류 초과({nutrition_100g.get('당류', 0):.1f}g)")
        reasons['저탄고지'] = ", ".join(failed_conditions) if failed_conditions else "모든 조건 만족"
        

        # === 저염 식단 분류 ===
        sodium_limit = self.classification_rules['저염']['sodium_per_100g_max']
        sodium_check = nutrition_100g.get('sodium_mg', 0) <= sodium_limit
        classifications['저염'] = sodium_check
        
        if sodium_check:
            reasons['저염'] = "조건 만족"
        else:
            reasons['저염'] = f"나트륨 초과({nutrition_100g.get('sodium_mg', 0):.1f}mg)"
        

        # === 채식 식단 분류 ===
        # 레시피명과 모든 재료명을 하나의 문자열로 합치기
        recipe_text = recipe_name + ' ' + ' '.join([d['original'] for d in nutrition_result['matching_details']])
        
        # 동물성 재료 키워드 확인
        exclude_keywords = self.classification_rules['채식']['exclude_keywords']
        
        # 리스트 컴프리헨션: 조건을 만족하는 요소들만 새 리스트로 생성
        found_non_veg = [keyword for keyword in exclude_keywords if keyword in recipe_text]
        
        # 동물성 재료가 없으면 채식
        is_vegetarian = len(found_non_veg) == 0
        classifications['채식'] = is_vegetarian
        
        if is_vegetarian:
            reasons['채식'] = "조건 만족"
        else:
            reasons['채식'] = f"동물성 재료: {', '.join(found_non_veg)}"
        
        # === 매칭률 계산 ===
        # 유사도가 0.8 이상인 매칭의 비율 계산
        high_similarity_matches = sum(1 for d in nutrition_result['matching_details'] 
                                    if d['similarity'] >= 0.8)
        
        # 조건부 표현식 (삼항 연산자): 조건 ? 참일때값 : 거짓일때값
        matching_rate = (high_similarity_matches / len(nutrition_result['matching_details']) * 100) \
                       if nutrition_result['matching_details'] else 0
        
        # 최종 결과 반환
        return {
            'recipe_name': recipe_name,
            'nutrition_result': nutrition_result,
            'classifications': classifications,
            'reasons': reasons,
            'matching_rate': matching_rate
        }
    

    '''테스트용. 지워도 되고 '''
    def print_result(self, result: Dict):
        # 분석 결과를 보기 좋게 출력하는 메서드
        # result: classify_recipe()에서 반환된 결과 딕셔너리
        
        print(f"\n{result['recipe_name']}")
        print("=" * 50)  # 구분선 출력
        
        nutrition_result = result['nutrition_result']
        nutrition_100g = nutrition_result['nutrition_per_100g']
        
        # 기본 정보 출력
        print(f"총 중량: {nutrition_result['total_weight_g']:.1f}g")
        print(f"재료 매칭률: {result['matching_rate']:.1f}%")
        
        # AI 사용 여부 표시
        ai_status = "AI 매칭" if self.use_ai else "기본 매칭"
        print(f"매칭 방식: {ai_status}")
        
        # 100g당 영양소 정보 출력 (국가표준식품성분표 컬럼명 사용)
        print(f"\n영양소 정보 (100g당):")
        print(f"   칼로리: {nutrition_100g.get('kcal', 0):.1f} kcal")
        print(f"   탄수화물: {nutrition_100g.get('carb_g', 0):.1f}g")
        print(f"   단백질: {nutrition_100g.get('protein_g', 0):.1f}g")
        print(f"   지방: {nutrition_100g.get('fat_g', 0):.1f}g")
        print(f"   나트륨: {nutrition_100g.get('sodium_mg', 0):.1f}mg")
        print(f"   당류: {nutrition_100g.get('당류', 0):.1f}g")  # 국가표준 컬럼명
        print(f"   식이섬유: {nutrition_100g.get('총  식이섬유', 0):.1f}g")  # 국가표준 컬럼명 (공백 포함)
        
        # 분류 결과 출력
        print(f"\n분류 결과:")
        for label, is_classified in result['classifications'].items():
            # 조건부 표현식으로 O/X 표시
            status = "O" if is_classified else "X"
            reason = result['reasons'][label]
            print(f"   {status} {label}: {reason}")
        
        # 적합한 식단 유형 출력
        # 리스트 컴프리헨션으로 True인 라벨들만 필터링
        applicable_labels = [label for label, classified in result['classifications'].items() if classified]
        
        if applicable_labels:
            print(f"\n적합한 식단: {', '.join(applicable_labels)}")
        else:
            print(f"\n적합한 식단: 일반식")



'''이것도 테스트용이긴 한데 지울려면 요소 몇개는 뽑아가야함'''
def main():
    # 메인 실행 함수
    # 프로그램의 진입점으로 전체 실행 흐름을 제어
    
    print("AI 기반 레시피 분류기")
    print("=" * 50)
    
    # AI 사용 여부 설정
    # True: AI 매칭 사용, False: 기본 문자열 매칭 사용
    use_ai = True
    
    # 분류기 객체 생성 및 초기화
    # CSV 파일 경로와 AI 사용 여부를 매개변수로 전달
    classifier = SmartRecipeClassifier('전처리_국가표준식품성분표.csv', use_ai=use_ai)
    
    # 테스트용 레시피 데이터 정의
    # 다양한 재료명과 단위를 포함하여 AI 매칭 성능 테스트
    test_recipes = {
        # 테스트 1
        "치킨 샐러드": {
            "재료": [
                {"item": "닭 가슴살", "amount": 150, "unit": "g"},      # 공백이 포함된 재료명
                {"item": "상추잎", "amount": 80, "unit": "g"},          # 변형된 표현
                {"item": "방울토마토", "amount": 10, "unit": "개"},        # 개수 단위
                {"item": "오이", "amount": 50, "unit": "g"}
            ],
            "조미료": [
                {"item": "엑스트라 버진 올리브오일", "amount": 1, "unit": "큰술"},  # 긴 이름의 재료
                {"item": "천일염", "amount": 1, "unit": "조금"}               # 다른 소금 표현
            ]
        },
        
        # 테스트 2
        "한식 불고기": {
            "재료": [
                {"item": "소 등심", "amount": 200, "unit": "g"},       # 공백이 포함된 고기
                {"item": "양파", "amount": 1, "unit": "개"},
                {"item": "당근", "amount": 50, "unit": "g"},
                {"item": "파", "amount": 2, "unit": "줄기"}
            ],
            "조미료": [
                {"item": "진간장", "amount": 3, "unit": "큰술"},
                {"item": "매실청", "amount": 2, "unit": "큰술"},       # AI가 설탕류로 매칭할지 테스트
                {"item": "참기름", "amount": 1, "unit": "작은술"},
                {"item": "다진 마늘", "amount": 1, "unit": "큰술"}      # 형용사가 포함된 재료명
            ]
        }
    }
    
    # 각 레시피에 대해 분류 실행
    # items() 메서드: 딕셔너리의 (키, 값) 쌍을 튜플로 반환
    for recipe_name, recipe_data in test_recipes.items():
        # 레시피 분류 수행
        result = classifier.classify_recipe(recipe_name, recipe_data)
        # 결과 출력
        classifier.print_result(result)




if __name__ == "__main__":
    main()