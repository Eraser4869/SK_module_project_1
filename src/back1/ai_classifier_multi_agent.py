# 멀티에이전트 기반 레시피 분류 시스템
# 각 에이전트가 전문 역할을 담당하고 서로 협업하는 구조
# sentence-transformers를 사용한 의미적 유사도 매칭 시스템

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from difflib import SequenceMatcher  # 문자열 유사도 계산용 (기본 매칭에서 사용)
from abc import ABC, abstractmethod  # 추상 기본 클래스 정의용

# AI 라이브러리 import (설치 필요: pip install sentence-transformers torch)
# try-except 구문을 사용하여 라이브러리 설치 여부 확인
try:
    from sentence_transformers import SentenceTransformer, util  # 문장 임베딩 모델
    import torch  # PyTorch 딥러닝 프레임워크
    AI_AVAILABLE = True  # AI 사용 플래그
except ImportError:
    # 라이브러리가 설치되지 않은 경우 기본 매칭으로 폴백
    AI_AVAILABLE = False



# 모든 에이전트의 기본 클래스 - 추상 클래스로 공통 기능 정의
class BaseAgent(ABC):

    def __init__(self, agent_name: str):
        # 에이전트 이름 설정
        self.agent_name = agent_name
        # 다른 에이전트들과의 연결을 저장할 딕셔너리
        self.other_agents = {}
    

    # 다른 에이전트를 등록하여 협업
    def register_agent(self, agent_name: str, agent_instance):
        """        
        Args:
            agent_name: 등록할 에이전트의 이름
            agent_instance: 에이전트 객체 인스턴스
        """
        # 딕셔너리에 에이전트 등록
        self.other_agents[agent_name] = agent_instance
    

    # 다른 에이전트의 메서드를 호출 (툴처럼 사용)
    def call_agent(self, agent_name: str, method_name: str, *args, **kwargs):
        """        
        Args:
            agent_name: 호출할 에이전트 이름
            method_name: 호출할 메서드 이름
            *args: 위치 인수들
            **kwargs: 키워드 인수들
            
        Returns:
            호출된 메서드의 반환값
        """
        # 에이전트가 등록되어 있는지 확인
        if agent_name not in self.other_agents:
            raise ValueError(f"{agent_name} 에이전트가 등록되지 않았습니다.")
        
        # 등록된 에이전트 객체 가져오기
        agent = self.other_agents[agent_name]
        # getattr(): 객체에서 메서드를 동적으로 가져옴
        method = getattr(agent, method_name)
        
        # 메서드 호출하고 결과 반환
        return method(*args, **kwargs)
    

    # 각 에이전트의 주요 처리 로직 - 하위 클래스에서 반드시 구현 필요
    @abstractmethod
    def process(self, *args, **kwargs):
        pass




# 데이터 관리 전용 에이전트 - 영양소 DB와 기본 데이터 관리
class DataManagerAgent(BaseAgent):
    
    def __init__(self, nutrition_csv_path: str):
        # 부모 클래스 초기화 호출
        super().__init__("DataManager")
        
        # print() 함수로 진행 상황 표시
        print("데이터 로딩 중...")
        # private 메서드를 호출하여 데이터 초기화
        self.nutrition_db = self._load_nutrition_data(nutrition_csv_path)
        self.fallback_nutrition = self._create_fallback_nutrition()
        self.unit_conversion = self._create_unit_conversion()
        # f-string으로 완료 메시지와 DB 크기 표시
        print(f"데이터 로딩 완료 (DB: {len(self.nutrition_db)}개 항목)")
    

    # 데이터 관련 요청 관리
    def process(self, action: str, **kwargs):
        """       
        Args:
            action: 수행할 작업 종류 (문자열)
            **kwargs: 작업에 필요한 추가 매개변수들
            
        Returns:
            요청된 데이터
        """

    
        # if-elif-else 구조로 액션 분기 처리
        if action == "get_nutrition":
            # get() 메서드: 딕셔너리에서 키값 안전하게 가져오기
            return self.get_nutrition_data(kwargs.get('item_name'))
        elif action == "get_fallback":
            return self.fallback_nutrition
        elif action == "get_conversion":
            return self.unit_conversion
        elif action == "get_all_items":
            # list() 함수로 pandas Index를 일반 리스트로 변환 후 기본 재료와 합치기
            return list(self.nutrition_db.index) + list(self.fallback_nutrition.keys())
        else:
            # 예외 발생시키기
            raise ValueError(f"알 수 없는 액션: {action}")
    

    # 특정 재료의 영양소 데이터 반환
    def get_nutrition_data(self, item_name: str) -> Optional[Dict]:
        """        
        Args:
            item_name: 조회할 재료명
            
        Returns:
            영양소 정보 딕셔너리 또는 None
        """
        # in 연산자로 pandas Index에서 재료명 존재 여부 확인
        if item_name in self.nutrition_db.index:
            # .loc[] 인덱서로 행 데이터 가져오고 .to_dict()로 딕셔너리 변환
            return self.nutrition_db.loc[item_name].to_dict()
        # elif로 fallback 데이터에서도 확인
        elif item_name in self.fallback_nutrition:
            return self.fallback_nutrition[item_name]
        # 둘 다 없으면 None 반환
        return None
    

    # 국가표준식품성분표 파일을 로드하는 메서드
    def _load_nutrition_data(self, csv_path: str) -> pd.DataFrame:
        """        
        Args:
            csv_path: CSV 파일의 경로
            
        Returns:
            재료명(item)을 인덱스로 하는 DataFrame
        """
        # pandas.read_csv()로 CSV 파일 읽기, encoding 매개변수로 한글 처리
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # dropna() 메서드로 'item' 컬럼이 비어있는 행 제거
        # subset 매개변수: 특정 컬럼들만 확인
        df = df.dropna(subset=['item'])
        
        # set_index() 메서드로 'item' 컬럼을 인덱스로 설정
        # inplace=True: 원본 DataFrame 수정
        df.set_index('item', inplace=True)
        
        return df
    

    # 기본 영양소 데이터를 정의하는 비상용 메서드
    def _create_fallback_nutrition(self) -> Dict[str, Dict]:
        """
        Returns:
            재료명을 키로 하고 영양소 정보를 값으로 하는 중첩 딕셔너리
        """
        
        # 중첩 딕셔너리 구조: {재료명: {영양소명: 값}}
        return {
            # 기본 조미료들 (100g 당 영양소)
            '소금': {
                'kcal': 0, 'carb_g': 0, 'fat_g': 0, 'protein_g': 0, 
                'sodium_mg': 40000, '당류': 0, '총  식이섬유': 0
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
    


    # 단위 변환 테이블을 생성하는 메서드
    def _create_unit_conversion(self) -> Dict:
        """        
        Returns:
            변환 정보를 담은 중첩 딕셔너리
        """
        # 중첩 딕셔너리로 변환 정보 구성
        return {
            # 기본 단위 변환 비율 (그램 기준)
            'conversion_rates': {
                'g': 1.0, 'kg': 1000.0, 'mg': 0.001, 'ml': 1.0, 'cc': 1.0, 'l': 1000.0,
                '큰술': 15.0, '작은술': 5.0, '컵': 200.0, '국자': 50.0
            },
            # 개수 단위를 그램으로 변환 (재료별 평균 무게)
            'piece_weights': {
                '계란': {'개': 50, '알': 50}, '마늘': {'쪽': 3, '개': 20}, '양파': {'개': 200},
                '토마토': {'개': 150}, '감자': {'개': 150}, '두부': {'모': 300}, '김': {'장': 3}
            },
            # 재료별 밀도 (부피를 무게로 변환시 사용)
            'densities': {
                '물': 1.0, '간장': 1.15, '참기름': 0.92, '올리브오일': 0.92, '밀가루': 0.6,
                '설탕': 0.8, '소금': 1.2, '고춧가루': 0.3
            }
        }



# 재료 매칭 전용 에이전트 - AI 기반 재료명 매칭
class IngredientMatchingAgent(BaseAgent):

    
    def __init__(self, use_ai: bool = True):
        # 부모 클래스 초기화
        super().__init__("IngredientMatcher")
        
        # and 연산자로 두 조건 모두 만족해야 AI 사용
        self.use_ai = use_ai and AI_AVAILABLE
        # AI 초기화 상태를 추적하는 플래그
        self.is_ai_initialized = False
    
    # 에이전트 등록 후 AI 초기화 - 지연 초기화 패턴
    def initialize_ai_after_registration(self):
        # and 연산자로 조건 체크: AI 사용 설정 & 아직 초기화 안됨
        if self.use_ai and not self.is_ai_initialized:
            # private 메서드 호출
            self._initialize_ai_matcher()
            # 플래그 업데이트
            self.is_ai_initialized = True
    

    # 재료 매칭 처리
    def process(self, ingredient_name: str, **kwargs) -> Tuple[str, float, Dict]:
        """        
        Args:
            ingredient_name: 매칭할 재료명
            **kwargs: 추가 매개변수들
            
        Returns:
            (매칭된 재료명, 유사도, 영양소 정보) 튜플
        """
        # AI 초기화가 필요한 경우 지연 초기화 수행
        if self.use_ai and not self.is_ai_initialized:
            self.initialize_ai_after_registration()
        
        # 조건부 분기로 매칭 방식 선택
        if self.use_ai:
            return self._match_with_ai(ingredient_name)
        else:
            return self._match_basic(ingredient_name)
    
    
    # AI 매칭 시스템을 초기화하는 메서드
    def _initialize_ai_matcher(self):

        # 한국어 특화 임베딩 모델 로드
         # jhgan/ko-sbert-nli: Hugging face에서 가져온 문장 임베딩 학습 모델. 한국어는 이게 제일 잘되는듯...
        model_name = 'jhgan/ko-sbert-nli'
        self.embedding_model = SentenceTransformer(model_name)
        
        # 캐시 파일 경로 설정 (임베딩 재계산 방지)
        self.embeddings_cache_path = 'nutrition_embeddings.pt'
        self.items_cache_path = 'nutrition_items.json'
        
        # DataManager에서 전체 재료 목록 가져오기 (에이전트 간 협업)
        self.food_items = self.call_agent("DataManager", "process", "get_all_items")
        
        # 임베딩 로드 또는 새로 생성
        self._load_or_create_embeddings()
    


    # 임베딩 캐시를 로드하거나 새로 생성하는 메서드
    def _load_or_create_embeddings(self):

        # os.path.exists()로 파일 존재 여부 확인, and 연산자로 두 파일 모두 체크
        if (os.path.exists(self.embeddings_cache_path) and 
            os.path.exists(self.items_cache_path)):
            
            # with 구문으로 파일 안전하게 열기/닫기
            with open(self.items_cache_path, 'r', encoding='utf-8') as f:
                # json.load()로 JSON 파일에서 리스트 읽기
                cached_items = json.load(f)
            
            # == 연산자로 리스트 동등성 비교
            if cached_items == self.food_items:
                # torch.load()로 PyTorch 텐서 파일 로드
                self.food_embeddings = torch.load(self.embeddings_cache_path)
                return
        
        # 새로운 임베딩 생성 (최초 실행시)
        # encode() 메서드로 텍스트 리스트를 고차원 벡터로 변환
        self.food_embeddings = self.embedding_model.encode(
            self.food_items,            # 인코딩할 텍스트 리스트
            convert_to_tensor=True,     # PyTorch 텐서로 변환
            show_progress_bar=True      # 진행률 표시
        )
        
        # torch.save()로 생성된 임베딩을 파일에 저장
        torch.save(self.food_embeddings, self.embeddings_cache_path)
        
        # with 구문으로 JSON 파일 쓰기
        with open(self.items_cache_path, 'w', encoding='utf-8') as f:
            # json.dump()으로 리스트를 JSON 파일로 저장
            # ensure_ascii=False: 한글 그대로 저장, indent=2: 보기 좋게 들여쓰기
            json.dump(self.food_items, f, ensure_ascii=False, indent=2)
    

    # AI 기반 재료 매칭을 수행하는 메서드
    def _match_with_ai(self, ingredient_name: str, threshold: float = 0.6) -> Tuple[str, float, Dict]:
        """        
        Args:
            ingredient_name: 찾을 재료명
            threshold: 유사도 임계값 (기본 0.6)
            
        Returns:
            (매칭된 재료명, 유사도, 영양소 정보) 튜플
        """
        # strip() 메서드로 앞뒤 공백 제거
        ingredient_name = ingredient_name.strip()
        
        # 1단계: in 연산자로 정확한 매칭 확인
        if ingredient_name in self.food_items:
            # 다른 에이전트 호출하여 영양소 데이터 가져오기
            nutrition = self.call_agent("DataManager", "get_nutrition_data", ingredient_name)
            return ingredient_name, 1.0, nutrition
        
        # 2단계: AI 의미적 유사도 계산
        # encode() 메서드로 입력 재료명을 임베딩 벡터로 변환
        query_embedding = self.embedding_model.encode(ingredient_name, convert_to_tensor=True)
        
        # util.cos_sim()으로 코사인 유사도 계산 (의미적 유사성 측정)
        # [0]으로 첫 번째 결과만 가져오기 (쿼리가 1개이므로)
        similarities = util.cos_sim(query_embedding, self.food_embeddings)[0]
        
        # torch.argmax()로 가장 높은 유사도의 인덱스 찾기
        # .item()으로 텐서에서 스칼라 값 추출
        best_idx = torch.argmax(similarities).item()
        best_similarity = similarities[best_idx].item()
        
        # >= 연산자로 임계값과 비교
        if best_similarity >= threshold:
            # 리스트 인덱싱으로 해당 재료명 가져오기
            best_match = self.food_items[best_idx]
            # 다른 에이전트에서 영양소 데이터 가져오기
            nutrition = self.call_agent("DataManager", "get_nutrition_data", best_match)
            return best_match, best_similarity, nutrition
        
        # 3단계: 매칭 실패시 기본값 반환 (물로 대체)
        fallback_nutrition = self.call_agent("DataManager", "process", "get_fallback")
        return '물', best_similarity, fallback_nutrition['물']
    


    # 기본 문자열 매칭을 수행하는 메서드 (AI 미사용시)
    def _match_basic(self, ingredient_name: str) -> Tuple[str, float, Dict]:
        """
        Args:
            ingredient_name: 찾을 재료명
            
        Returns:
            (매칭된 재료명, 유사도, 영양소 정보) 튜플
        """
        # strip() 메서드로 공백 제거
        ingredient_name = ingredient_name.strip()
        
        # 1단계: 정확한 매칭 확인
        nutrition = self.call_agent("DataManager", "get_nutrition_data", ingredient_name)
        if nutrition:  # None이 아니면 True
            return ingredient_name, 1.0, nutrition
        
        # 2단계: 부분 매칭 수행
        all_items = self.call_agent("DataManager", "process", "get_all_items")
        best_match = None
        best_score = 0
        
        # for 루프로 모든 재료와 비교
        for db_name in all_items:
            # in 연산자로 부분 문자열 포함 관계 확인 (양방향)
            if ingredient_name in db_name or db_name in ingredient_name:
                # SequenceMatcher로 문자열 유사도 계산 (0~1 범위)
                # .lower() 메서드로 대소문자 무시
                score = SequenceMatcher(None, ingredient_name.lower(), db_name.lower()).ratio()
                # > 연산자로 더 높은 점수 찾기
                if score > best_score:
                    best_score = score
                    best_match = db_name
        
        # and 연산자로 두 조건 모두 만족시 매칭 성공
        if best_match and best_score > 0.6:
            nutrition = self.call_agent("DataManager", "get_nutrition_data", best_match)
            return best_match, best_score, nutrition
        
        # 3단계: 최종 기본값 (물)
        fallback_nutrition = self.call_agent("DataManager", "process", "get_fallback")
        return '물', 0.1, fallback_nutrition['물']


# 단위 변환 전용 에이전트
class UnitConversionAgent(BaseAgent):
    
    def __init__(self):
        # 부모 클래스 초기화
        super().__init__("UnitConverter")
    

    # 단위를 그램으로 변환
    def process(self, amount: float, unit: str, ingredient_name: str = '') -> float:
        """        
        Args:
            amount: 양 (숫자)
            unit: 단위 (문자열)
            ingredient_name: 재료명 (단위 변환시 참고용, 기본값 빈 문자열)
            
        Returns:
            그램 단위로 변환된 양
        """

        # DataManager에서 변환 테이블 가져오기 (에이전트 간 협업)
        conversion_data = self.call_agent("DataManager", "process", "get_conversion")
        
        # .lower().strip() 체이닝으로 소문자 변환 후 공백 제거
        unit = unit.lower().strip()
        ingredient_name = ingredient_name.lower().strip()
        
        # 1단계: in 연산자로 기본 무게/부피 단위 확인
        if unit in conversion_data['conversion_rates']:
            # 딕셔너리 접근으로 변환 비율 가져와서 곱셈 연산
            base_amount = amount * conversion_data['conversion_rates'][unit]
            
            # in 연산자로 부피 단위인지 확인
            if unit in ['ml', 'cc', 'l', '큰술', '작은술', '컵']:
                # .items() 메서드로 딕셔너리의 (키, 값) 쌍 순회
                for ingredient, density in conversion_data['densities'].items():
                    # in 연산자로 재료명 포함 여부 확인
                    if ingredient in ingredient_name:
                        # 부피 × 밀도 = 무게 계산
                        return base_amount * density
                # 기본 밀도 1.0 (물 기준) 적용
                return base_amount
            
            return base_amount
        
        # 2단계: 개수 단위 변환
        # .items() 메서드로 중첩 딕셔너리 순회
        for ingredient, weights in conversion_data['piece_weights'].items():
            # and 연산자로 두 조건 모두 만족하는지 확인
            if ingredient in ingredient_name and unit in weights:
                # 개수 × 개당 무게 계산
                return amount * weights[unit]
        
        # 3단계: 기본 개수 단위 (평균값 사용)
        default_weights = {'개': 100, '알': 50, '쪽': 3, '장': 3, '모': 300}
        if unit in default_weights:
            return amount * default_weights[unit]
        
        # 4단계: 변환 불가능한 경우 원래 값 그대로 반환
        return amount



# 영양소 계산 전용 에이전트
class NutritionCalculatorAgent(BaseAgent):
    
    def __init__(self):
        # 부모 클래스 초기화
        super().__init__("NutritionCalculator")
    

    # 레시피의 전체 영양소 계산
    def process(self, recipe_data: Dict) -> Dict:
        """        
        Args:
            recipe_data: 레시피 정보가 담긴 딕셔너리
            
        Returns:
            영양소 계산 결과를 담은 딕셔너리
        """
        # 딕셔너리 리터럴로 총 영양소 초기화
        total_nutrition = {
            'kcal': 0, 'carb_g': 0, 'fat_g': 0, 'protein_g': 0, 
            'sodium_mg': 0, '당류': 0, '총  식이섬유': 0
        }
        
        # 숫자 변수 초기화
        total_weight_g = 0
        # 빈 리스트로 매칭 상세 정보 초기화
        matching_details = []
        
        # .get() 메서드로 키가 없을 때 빈 리스트 반환, + 연산자로 리스트 합치기
        all_ingredients = recipe_data.get('재료', []) + recipe_data.get('조미료', [])
        
        # for 루프로 각 재료 처리
        for ingredient in all_ingredients:
            # .get() 메서드로 딕셔너리에서 값 안전하게 가져오기
            item_name = ingredient.get('item', '')
            amount = ingredient.get('amount', 0)
            unit = ingredient.get('unit', 'g')
            
            # not 연산자와 or 연산자로 유효성 검사
            # <= 연산자로 숫자 비교
            if not item_name or amount <= 0:
                continue  # 다음 반복으로 건너뛰기
            
            # 다른 에이전트들에게 작업 위임 (에이전트 간 협업)
            matched_name, similarity, nutrition = self.call_agent(
                "IngredientMatcher", "process", item_name
            )
            
            weight_g = self.call_agent(
                "UnitConverter", "process", amount, unit, item_name
            )
            
            # += 연산자로 총 중량에 누적
            total_weight_g += weight_g
            # 나눗셈 연산으로 100g 대비 실제 사용량 비율 계산
            multiplier = weight_g / 100
            
            # for 루프로 각 영양소별 계산
            for nutrient in total_nutrition.keys():
                # and 연산자로 두 조건 모두 확인: 영양소 존재 & None이 아님
                if nutrient in nutrition and nutrition[nutrient] is not None:
                    # pandas.isna()로 NaN 값 체크, not 연산자로 반전
                    if not pd.isna(nutrition[nutrient]):
                        # += 연산자로 영양소 누적 (곱셈으로 실제 사용량 반영)
                        total_nutrition[nutrient] += nutrition[nutrient] * multiplier
            
            # .append() 메서드로 리스트에 딕셔너리 추가
            matching_details.append({
                'original': item_name,
                'matched': matched_name,
                'similarity': similarity,
                'weight_g': weight_g
            })
        
        # 빈 딕셔너리 초기화
        nutrition_per_100g = {}
        # > 연산자로 0으로 나누기 방지
        if total_weight_g > 0:
            # .items() 메서드로 딕셔너리 순회
            for nutrient, total_value in total_nutrition.items():
                # (총 영양소 / 총 중량) × 100 = 100g당 영양소 계산
                nutrition_per_100g[nutrient] = (total_value / total_weight_g) * 100
        
        # 딕셔너리 리터럴로 결과 반환
        return {
            'total_nutrition': total_nutrition,
            'nutrition_per_100g': nutrition_per_100g,
            'total_weight_g': total_weight_g,
            'matching_details': matching_details
        }



# 레시피 분류 전용 에이전트
class RecipeClassificationAgent(BaseAgent):
    
    def __init__(self):
        # 부모 클래스 초기화
        super().__init__("RecipeClassifier")
        # private 메서드 호출하여 분류 규칙 초기화
        self.classification_rules = self._define_classification_rules()
    

    # 레시피 분류 수행
    def process(self, recipe_name: str, nutrition_result: Dict) -> Dict:
        """        
        Args:
            recipe_name: 레시피 이름
            nutrition_result: 영양소 계산 결과
            
        Returns:
            분류 결과를 담은 딕셔너리
        """
        # 딕셔너리 접근으로 100g당 영양소 데이터 가져오기
        nutrition_100g = nutrition_result['nutrition_per_100g']
        # 빈 딕셔너리들 초기화
        classifications = {}
        reasons = {}
        
        # === 다이어트 식단 분류 ===
        # 딕셔너리 접근으로 분류 규칙 가져오기
        diet_rules = self.classification_rules['다이어트']
        
        # 리스트 컴프리헨션으로 각 조건 체크
        # .get() 메서드로 기본값 0 설정, <= >= 연산자로 조건 비교
        diet_checks = [
            nutrition_100g.get('kcal', 0) <= diet_rules['kcal_per_100g_max'],
            nutrition_100g.get('fat_g', 0) <= diet_rules['fat_per_100g_max'],
            nutrition_100g.get('protein_g', 0) >= diet_rules['protein_per_100g_min'],
            nutrition_100g.get('당류', 0) <= diet_rules['sugar_per_100g_max']
        ]
        
        # all() 함수로 모든 조건이 True인지 확인
        classifications['다이어트'] = all(diet_checks)
        
        # 빈 리스트 초기화
        failed_conditions = []
        # not 연산자로 조건 실패 확인
        if not diet_checks[0]: 
            # f-string으로 문자열 포맷팅, :.1f로 소수점 1자리 표시
            failed_conditions.append(f"칼로리 초과({nutrition_100g.get('kcal', 0):.1f}kcal)")
        if not diet_checks[1]: 
            failed_conditions.append(f"지방 초과({nutrition_100g.get('fat_g', 0):.1f}g)")
        if not diet_checks[2]: 
            failed_conditions.append(f"단백질 부족({nutrition_100g.get('protein_g', 0):.1f}g)")
        if not diet_checks[3]: 
            failed_conditions.append(f"당류 초과({nutrition_100g.get('당류', 0):.1f}g)")
        
        # 조건부 표현식(삼항 연산자): 조건 ? 참일때값 : 거짓일때값
        # .join() 메서드로 리스트 요소들을 문자열로 연결
        reasons['다이어트'] = ", ".join(failed_conditions) if failed_conditions else "모든 조건 만족"
        
        # === 저탄고지(케토) 식단 분류 ===
        keto_rules = self.classification_rules['저탄고지']
        keto_checks = [
            nutrition_100g.get('carb_g', 0) <= keto_rules['carb_per_100g_max'],
            nutrition_100g.get('fat_g', 0) >= keto_rules['fat_per_100g_min'],
            nutrition_100g.get('protein_g', 0) >= keto_rules['protein_per_100g_min'],
            nutrition_100g.get('당류', 0) <= keto_rules['sugar_per_100g_max']
        ]
        classifications['저탄고지'] = all(keto_checks)
        
        # 실패 조건 체크 (다이어트와 동일한 패턴)
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
        # 딕셔너리 중첩 접근으로 나트륨 제한값 가져오기
        sodium_limit = self.classification_rules['저염']['sodium_per_100g_max']
        # <= 연산자로 나트륨 기준 확인
        sodium_check = nutrition_100g.get('sodium_mg', 0) <= sodium_limit
        classifications['저염'] = sodium_check
        
        # 조건부 표현식으로 이유 설정
        if sodium_check:
            reasons['저염'] = "조건 만족"
        else:
            reasons['저염'] = f"나트륨 초과({nutrition_100g.get('sodium_mg', 0):.1f}mg)"
        
        # === 채식 식단 분류 ===
        # + 연산자로 문자열 연결, .join() 메서드로 리스트를 문자열로 변환
        # 리스트 컴프리헨션으로 매칭 상세 정보에서 원본 재료명만 추출
        recipe_text = recipe_name + ' ' + ' '.join([d['original'] for d in nutrition_result['matching_details']])
        
        # 딕셔너리 접근으로 제외 키워드 리스트 가져오기
        exclude_keywords = self.classification_rules['채식']['exclude_keywords']
        
        # 리스트 컴프리헨션으로 조건 필터링: for절과 if절 조합
        # in 연산자로 키워드가 텍스트에 포함되는지 확인
        found_non_veg = [keyword for keyword in exclude_keywords if keyword in recipe_text]
        
        # len() 함수로 리스트 길이 확인, == 연산자로 0과 비교
        is_vegetarian = len(found_non_veg) == 0
        classifications['채식'] = is_vegetarian
        
        # 조건부 표현식으로 이유 설정
        if is_vegetarian:
            reasons['채식'] = "조건 만족"
        else:
            reasons['채식'] = f"동물성 재료: {', '.join(found_non_veg)}"
        
        # === 매칭률 계산 ===
        # sum() 함수와 제네레이터 표현식으로 조건 만족 개수 계산
        # >= 연산자로 유사도 0.8 이상인 매칭 개수 세기
        high_similarity_matches = sum(1 for d in nutrition_result['matching_details'] 
                                    if d['similarity'] >= 0.8)
        
        # 조건부 표현식으로 0으로 나누기 방지
        # len() 함수로 전체 매칭 개수 확인, * 연산자로 백분율 계산
        matching_rate = (high_similarity_matches / len(nutrition_result['matching_details']) * 100) \
                       if nutrition_result['matching_details'] else 0
        
        # 딕셔너리 리터럴로 결과 반환
        return {
            'classifications': classifications,
            'reasons': reasons,
            'matching_rate': matching_rate
        }
    


    # 레시피 분류 규칙을 정의하는 메서드
    def _define_classification_rules(self) -> Dict:
        """        
        Returns:
            분류 규칙을 담은 중첩 딕셔너리
        """
        # 중첩 딕셔너리 구조로 각 식단별 기준값 정의
        return {
            # 다이어트 식단 기준 (저칼로리, 저지방, 고단백, 저당)
            '다이어트': {
                'kcal_per_100g_max': 250,
                'fat_per_100g_max': 10,
                'protein_per_100g_min': 8,
                'sugar_per_100g_max': 15
            },
            # 저탄고지(케토) 식단 기준 (저탄수화물, 고지방, 적정단백질)
            '저탄고지': {
                'carb_per_100g_max': 15,
                'fat_per_100g_min': 10,
                'protein_per_100g_min': 10,
                'sugar_per_100g_max': 8
            },
            # 저염 식단 기준
            '저염': {
                'sodium_per_100g_max': 400
            },
            # 채식 식단 기준 (동물성 재료 제외)
            '채식': {
                # 리스트로 동물성 재료 키워드들 정의
                'exclude_keywords': [
                    '돼지', '소', '닭', '오리', '양', '생선', '연어', '참치', '새우', '게', 
                    '조개', '굴', '문어', '오징어', '멸치', '삼겹살', '갈비', '치킨', 
                    '햄', '소시지', '베이컨'
                ]
            }
        }



# 전체 워크플로우를 조율하는 마스터 에이전트
class CoordinatorAgent(BaseAgent):

    
    def __init__(self, nutrition_csv_path: str, use_ai: bool = True):
        # 부모 클래스 초기화
        super().__init__("Coordinator")
        
        # print() 함수로 멀티에이전트 시스템 초기화 시작 알림
        print("멀티에이전트 시스템 초기화 중...")
        # 모든 전문 에이전트들을 초기화 (AI는 지연 초기화)
        self.data_manager = DataManagerAgent(nutrition_csv_path)
        self.ingredient_matcher = IngredientMatchingAgent(use_ai)
        self.unit_converter = UnitConversionAgent()
        self.nutrition_calculator = NutritionCalculatorAgent()
        self.recipe_classifier = RecipeClassificationAgent()
        
        # print() 함수로 에이전트 네트워크 설정 상태 표시
        print("에이전트 네트워크 설정 중...")
        # private 메서드 호출하여 에이전트 간 연결 설정
        self._setup_agent_network()
        
        # and 연산자로 두 조건 모두 확인 후 AI 초기화
        if use_ai and AI_AVAILABLE:
            self.ingredient_matcher.initialize_ai_after_registration()
        
        # print() 함수로 시스템 준비 완료 알림
        print("멀티에이전트 시스템 준비 완료!")
    
    def _setup_agent_network(self):
        """에이전트들 간의 네트워크 연결 설정"""
        # 딕셔너리 리터럴로 에이전트들 정리
        agents = {
            "DataManager": self.data_manager,
            "IngredientMatcher": self.ingredient_matcher,
            "UnitConverter": self.unit_converter,
            "NutritionCalculator": self.nutrition_calculator,
            "RecipeClassifier": self.recipe_classifier
        }
        
        # 중첩 for 루프로 모든 에이전트가 서로를 알 수 있도록 등록
        # .items() 메서드로 딕셔너리의 (키, 값) 쌍 순회
        for agent_name, agent_instance in agents.items():
            for other_name, other_instance in agents.items():
                # != 연산자로 자기 자신은 제외
                if agent_name != other_name:
                    # 다른 에이전트 등록
                    agent_instance.register_agent(other_name, other_instance)
    


    # 전체 레시피 분석 워크플로우 조율
    def process(self, recipe_name: str, recipe_data: Dict) -> Dict:
        """        
        Args:
            recipe_name: 레시피 이름
            recipe_data: 레시피 데이터 (재료, 조미료 정보)
            
        Returns:
            최종 분석 결과를 담은 딕셔너리
        """
        # 1단계: NutritionCalculator 에이전트에게 영양소 계산 위임
        # (내부적으로 IngredientMatcher, UnitConverter 에이전트들과 협업)
        nutrition_result = self.nutrition_calculator.process(recipe_data)
        
        # 2단계: RecipeClassifier 에이전트에게 분류 작업 위임
        classification_result = self.recipe_classifier.process(recipe_name, nutrition_result)
        
        # 3단계: 딕셔너리 리터럴로 최종 결과 통합
        final_result = {
            'recipe_name': recipe_name,
            'nutrition_result': nutrition_result,
            'classifications': classification_result['classifications'],
            'reasons': classification_result['reasons'],
            'matching_rate': classification_result['matching_rate']
        }
        
        return final_result



# 멀티에이전트 레시피 시스템의 메인 클래스
class MultiAgentRecipeSystem:
    

    # 멀티에이전트 시스템 초기화
    def __init__(self, nutrition_csv_path: str, use_ai: bool = True):
        """        
        Args:
            nutrition_csv_path: 영양소 데이터 CSV 파일 경로
            use_ai: AI 매칭 사용 여부
        """
        # 코디네이터 에이전트 생성 (다른 모든 에이전트들을 관리)
        self.coordinator = CoordinatorAgent(nutrition_csv_path, use_ai)
    


    # 레시피 분석 (멀티에이전트 협업 워크플로우)
    def analyze_recipe(self, recipe_name: str, recipe_data: Dict) -> Dict:
        """
        Args:
            recipe_name: 레시피 이름
            recipe_data: 레시피 데이터 (재료, 조미료 정보)
            
        Returns:
            분석 결과 딕셔너리
        """
        # 코디네이터에게 전체 작업 위임
        return self.coordinator.process(recipe_name, recipe_data)
    

    # 각 에이전트의 상태 정보 반환
    def get_agent_status(self) -> Dict:
        """        
        Returns:
            에이전트 상태 정보를 담은 딕셔너리
        """
        # f-string을 사용한 문자열 포맷팅으로 상태 정보 구성
        return {
            'coordinator': self.coordinator.agent_name,
            'data_manager': f"{self.coordinator.data_manager.agent_name} (DB: {len(self.coordinator.data_manager.nutrition_db)}개 항목)",
            'ingredient_matcher': f"{self.coordinator.ingredient_matcher.agent_name} ({'AI' if self.coordinator.ingredient_matcher.use_ai else '기본'} 매칭)",
            'unit_converter': self.coordinator.unit_converter.agent_name,
            'nutrition_calculator': self.coordinator.nutrition_calculator.agent_name,
            'recipe_classifier': self.coordinator.recipe_classifier.agent_name
        }



# 메인 실행 함수 - 멀티에이전트 시스템 테스트
def main():
    
    # 멀티에이전트 시스템 초기화 (객체 생성)
    system = MultiAgentRecipeSystem('전처리_국가표준식품성분표.csv', use_ai=True)
    
    # 딕셔너리 리터럴로 테스트용 레시피 데이터 정의
    test_recipes = {
        "AI 매칭 테스트 - 치킨 샐러드": {
            "재료": [
                {"item": "닭 가슴살", "amount": 150, "unit": "g"},
                {"item": "상추잎", "amount": 80, "unit": "g"},
                {"item": "방울토마토", "amount": 10, "unit": "개"},
                {"item": "오이", "amount": 50, "unit": "g"}
            ],
            "조미료": [
                {"item": "엑스트라 버진 올리브오일", "amount": 1, "unit": "큰술"},
                {"item": "천일염", "amount": 1, "unit": "작은술"}
            ]
        },
        
        "복합 단위 테스트 - 한식 불고기": {
            "재료": [
                {"item": "소 등심", "amount": 200, "unit": "g"},
                {"item": "양파", "amount": 1, "unit": "개"},
                {"item": "당근", "amount": 50, "unit": "g"},
                {"item": "파", "amount": 30, "unit": "g"}
            ],
            "조미료": [
                {"item": "진간장", "amount": 3, "unit": "큰술"},
                {"item": "매실청", "amount": 2, "unit": "큰술"},
                {"item": "참기름", "amount": 1, "unit": "작은술"},
                {"item": "다진 마늘", "amount": 1, "unit": "큰술"}
            ]
        }
    }
    
    # .items() 메서드로 딕셔너리의 (키, 값) 쌍 순회
    for recipe_name, recipe_data in test_recipes.items():
        # 멀티에이전트 협업으로 레시피 분석
        result = system.analyze_recipe(recipe_name, recipe_data)
        
        # 분류 결과 출력
        print("분류 결과:")
        # .items() 메서드로 분류 결과 순회
        for label, is_classified in result['classifications'].items():
            # 조건부 표현식으로 O/X 표시
            status = "O" if is_classified else "X"
            reason = result['reasons'][label]
            print(f"  {status} {label}: {reason}")
        
        # 리스트 컴프리헨션으로 적합한 식단 필터링
        applicable_labels = [label for label, classified in result['classifications'].items() if classified]
        
        # 조건부 표현식으로 적합한 식단 출력
        if applicable_labels:
            # .join() 메서드로 리스트를 문자열로 연결
            print(f"적합한 식단: {', '.join(applicable_labels)}")
        else:
            print("적합한 식단: 일반식")
    
    # print() 함수로 전체 분석 완료 알림
    print(f"\n모든 레시피 분석 완료!")


# __name__ == "__main__" 조건문으로 스크립트 직접 실행시에만 main() 호출
if __name__ == "__main__":
    main()