# recipe_data_integrator.py - 올바른 방법

import json
from typing import Dict, List, Any, Tuple

# 기존 모듈들 import
from ai_classifier_multi_agent import MultiAgentRecipeSystem
from cooking_time_model import parse_json_and_predict

# ✅ 전역 변수로 실제 레시피 데이터 저장
REAL_RECIPE_DATA = None

def set_recipe_data(recipe_dict: Dict):
    """실제 레시피 데이터를 전역 변수에 설정"""
    global REAL_RECIPE_DATA
    REAL_RECIPE_DATA = recipe_dict
    print(f"✅ 실제 레시피 데이터 설정 완료: {len(recipe_dict)}개 레시피")
    print(f"📋 레시피 목록: {list(recipe_dict.keys())}")

class RecipeDataIntegrator:
    """
    레시피 데이터 통합 클래스
    - 식단 분류 모듈과 조리시간 예측 모듈의 결과를 통합
    - 사용자 선호도와 매칭할 수 있는 완전한 레시피 데이터 생성
    """
    
    def __init__(self, nutrition_csv_path: str = '전처리_국가표준식품성분표.csv', use_ai: bool = True):
        """
        Args:
            nutrition_csv_path: 영양소 데이터 CSV 파일 경로 (식단 분류용)
            use_ai: AI 매칭 사용 여부
        """
        print("레시피 데이터 통합기 초기화 중...")
        
        # 식단 분류 시스템 초기화
        self.diet_classifier = MultiAgentRecipeSystem(nutrition_csv_path, use_ai)
        
        # 선호도 매핑 정의
        self.preference_mapping = {
            'diet': ['다이어트', '저탄고지', '저염', '채식'],
            'time': ['15분', '30분', '45분'],
            'difficulty': ['쉬움', '보통', '어려움']
        }
        
        print("레시피 데이터 통합기 준비 완료!")
    
    def integrate_recipe_data(self, test_recipes: Dict, preferences: Dict = None) -> Tuple[Dict, Dict]:
        """
        테스트 레시피들의 식단 분류 결과와 조리시간 예측 결과를 통합
        
        Args:
            test_recipes: 분석할 레시피 딕셔너리
            preferences: 사용자 선호도 (선택적)
            
        Returns:
            (통합된 레시피 데이터 딕셔너리, 사용자 선호도 딕셔너리) 튜플
        """
        print(f"총 {len(test_recipes)} 개 레시피 데이터 통합 시작...")
        
        # 기본 선호도 설정 (모든 값이 False)
        if preferences is None:
            preferences = {
                "diet": [0, 0, 0, 0],
                "time": [0, 0, 0],
                "difficulty": [0, 0, 0]
            }
        
        # 1. 식단 분류 결과 가져오기
        diet_results = self._get_diet_classification_results(test_recipes)
        
        # 2. 조리시간 예측 결과 가져오기  
        cooking_time_results = self._get_cooking_time_results(test_recipes, preferences)
        
        # 3. 두 결과를 통합하고 선호도 매칭 점수 계산
        integrated_data = self._merge_results_with_preferences(test_recipes, diet_results, cooking_time_results, preferences)
        
        print(f"레시피 데이터 통합 완료!")
        return integrated_data, preferences
    
    def _get_diet_classification_results(self, test_recipes: Dict) -> Dict:
        """식단 분류 모듈에서 결과 가져오기"""
        print("식단 분류 결과 수집 중...")
        
        diet_results = {}
        for recipe_name, recipe_data in test_recipes.items():
            try:
                # ai_classifier_multi_agent 모듈의 결과 가져오기
                result = self.diet_classifier.analyze_recipe(recipe_name, recipe_data)
                
                diet_results[recipe_name] = {
                    'classifications': result['classifications'],
                    'reasons': result['reasons'],
                    'matching_rate': result['matching_rate'],
                    'nutrition_info': {
                        'total_nutrition': result['nutrition_result']['total_nutrition'],
                        'nutrition_per_100g': result['nutrition_result']['nutrition_per_100g'],
                        'total_weight_g': result['nutrition_result']['total_weight_g']
                    }
                }
                
            except Exception as e:
                print(f"식단 분류 오류 - {recipe_name}: {e}")
                diet_results[recipe_name] = {
                    'classifications': {'다이어트': False, '저탄고지': False, '저염': False, '채식': False},
                    'reasons': {'다이어트': '분석 실패', '저탄고지': '분석 실패', '저염': '분석 실패', '채식': '분석 실패'},
                    'matching_rate': 0.0,
                    'nutrition_info': None
                }
        
        return diet_results
    

    def _get_cooking_time_results(self, test_recipes: Dict, preferences: Dict) -> Dict:
        """조리시간 예측 모듈에서 결과 가져오기"""
        print("조리시간 예측 결과 수집 중...")
        
        try:
            # test_recipes를 JSON 문자열로 변환 (cooking_time_model 모듈이 요구하는 형식)
            json_str = json.dumps(test_recipes, ensure_ascii=False)
            
            # 선호도를 cooking_time_model이 요구하는 형식으로 변환
            preference_dict = self._convert_preferences_to_dict(preferences)
            
            # cooking_time_model 모듈의 결과 가져오기
            cooking_results = parse_json_and_predict(json_str, preference_dict)
            
            # 결과를 레시피명을 키로 하는 딕셔너리로 변환
            cooking_time_results = {}
            for result in cooking_results:
                recipe_name = result['recipe_name']
                cooking_time_results[recipe_name] = {
                    'cooking_kind': result.get('kind', '알 수 없음'),
                    'difficulty_level': result.get('level', '보통'),
                    'predicted_time_minutes': result.get('predicted_cooking_time', 30),
                    'time_category': self._categorize_time(result.get('predicted_cooking_time', 30)),
                    'ingredients_list': result.get('ingredients', [])
                }
            
            return cooking_time_results
            
        except Exception as e:
            print(f"조리시간 예측 오류: {e}")
            return {}
    


    def _convert_preferences_to_dict(self, preferences: Dict) -> Dict:
        """선호도를 cooking_time_model이 요구하는 형식으로 변환"""
        preference_dict = {
            "희망_식단": [],
            "희망_조리시간": [],
            "희망_난이도": []
        }
        
        # 식단 선호도 변환
        diet_prefs = preferences.get('diet', [0, 0, 0, 0])
        for i, pref in enumerate(diet_prefs):
            if pref == 1:
                preference_dict["희망_식단"].append(self.preference_mapping['diet'][i])
        
        # 조리시간 선호도 변환
        time_prefs = preferences.get('time', [0, 0, 0])
        for i, pref in enumerate(time_prefs):
            if pref == 1:
                preference_dict["희망_조리시간"].append(self.preference_mapping['time'][i])
        
        # 난이도 선호도 변환
        difficulty_prefs = preferences.get('difficulty', [0, 0, 0])
        for i, pref in enumerate(difficulty_prefs):
            if pref == 1:
                preference_dict["희망_난이도"].append(self.preference_mapping['difficulty'][i])
        
        return preference_dict
    
    def _categorize_time(self, minutes: float) -> str:
        """시간을 카테고리로 분류"""
        if minutes <= 15:
            return '15분'
        elif minutes <= 30:
            return '30분'
        else:
            return '45분'
    
    def _merge_results_with_preferences(self, test_recipes: Dict, diet_results: Dict, cooking_time_results: Dict, preferences: Dict) -> Dict:
        """식단 분류 결과와 조리시간 예측 결과를 통합하고 선호도 매칭 점수 계산"""
        print("결과 통합 및 선호도 매칭 점수 계산 중...")
        
        integrated_data = {}
        
        for recipe_name, recipe_data in test_recipes.items():
            # 기본 레시피 정보
            diet_result = diet_results.get(recipe_name, {})
            cooking_result = cooking_time_results.get(recipe_name, {})
            
            # 선호도 매칭 점수 계산
            preference_matches = self._calculate_preference_matching(diet_result, cooking_result, preferences)
            
            integrated_recipe = {
                'recipe_name': recipe_name,
                'original_recipe_data': recipe_data,
                
                # 식단 분류 결과
                'diet_classifications': diet_result.get('classifications', {}),
                'diet_reasons': diet_result.get('reasons', {}),
                'nutrition_info': diet_result.get('nutrition_info', {}),
                
                # 조리시간 예측 결과
                'cooking_info': cooking_result,
                
                # 선호도 매칭 결과
                'preference_matches': preference_matches,
                'preference_score': self._calculate_preference_score(preference_matches)
            }
            
            integrated_data[recipe_name] = integrated_recipe
        
        return integrated_data
    
    def _calculate_preference_matching(self, diet_result: Dict, cooking_result: Dict, preferences: Dict) -> Dict:
        """사용자 선호도와 매칭률 계산"""
        matches = {
            'diet_matches': [],
            'time_match': False,
            'difficulty_match': False
        }
        
        # 식단 선호도 매칭
        diet_prefs = preferences.get('diet', [0, 0, 0, 0])
        for i, pref in enumerate(diet_prefs):
            if pref == 1:
                diet_type = self.preference_mapping['diet'][i]
                if diet_result.get('classifications', {}).get(diet_type, False):
                    matches['diet_matches'].append(diet_type)
        
        # 조리시간 선호도 매칭
        time_prefs = preferences.get('time', [0, 0, 0])
        time_category = cooking_result.get('time_category', '30분')
        
        for i, pref in enumerate(time_prefs):
            if pref == 1 and self.preference_mapping['time'][i] == time_category:
                matches['time_match'] = True
                break
        
        # 난이도 선호도 매칭
        difficulty_prefs = preferences.get('difficulty', [0, 0, 0])
        difficulty_level = cooking_result.get('difficulty_level', '보통')
        
        for i, pref in enumerate(difficulty_prefs):
            if pref == 1 and self.preference_mapping['difficulty'][i] == difficulty_level:
                matches['difficulty_match'] = True
                break
        
        return matches
    
    def _calculate_preference_score(self, preference_matches: Dict) -> float:
        """선호도 매칭 점수 계산 (0-100점)"""
        score = 0
        
        # 식단 매칭 점수 (최대 80점)
        diet_matches = len(preference_matches.get('diet_matches', []))
        score += diet_matches * 20  # 식단 하나당 20점
        
        # 시간 매칭 점수 (최대 10점)
        if preference_matches.get('time_match', False):
            score += 10
        
        # 난이도 매칭 점수 (최대 10점)
        if preference_matches.get('difficulty_match', False):
            score += 10
        
        return min(score, 100)  # 최대 100점으로 제한


# 사용 예시 - 선호도도 함께 반환
def example_usage(user_preferences=None):
    """통합 모듈 사용 예시 - 실제 데이터 또는 하드코딩 데이터 사용"""
    
    print("example_usage() 함수 실행 시작...")
    
    # ✅ 전역 변수에 실제 데이터가 있으면 사용, 없으면 하드코딩 데이터 사용
    global REAL_RECIPE_DATA
    if REAL_RECIPE_DATA and isinstance(REAL_RECIPE_DATA, dict) and len(REAL_RECIPE_DATA) > 0:
        print(f"✅ 실제 레시피 데이터 사용: {len(REAL_RECIPE_DATA)}개")
        test_recipes = REAL_RECIPE_DATA
    else:
        print("⚠️ 실제 레시피 데이터가 없어 하드코딩 데이터 사용")
        # 하드코딩된 백업 데이터
        test_recipes = {
            "잡채": {
                "재료": [
                    {"item": "돼지고기", "amount": 100, "unit": "g"},
                    {"item": "당면", "amount": 50, "unit": "g"},
                    {"item": "시금치", "amount": 100, "unit": "g"}
                ],
                "조미료": [
                    {"item": "간장", "amount": 2, "unit": "큰술"},
                    {"item": "설탕", "amount": 1, "unit": "큰술"}
                ]
            }
        }
    
    preferences = user_preferences
    
    print(f"테스트 레시피 {len(test_recipes)}개 준비 완료")
    print(f"레시피 목록: {list(test_recipes.keys())}")
    print(f"선호도 설정 완료: {preferences}")
    
    try:
        # 통합기 초기화 및 실행
        integrator = RecipeDataIntegrator()
        integrated_data, user_preferences = integrator.integrate_recipe_data(test_recipes, preferences)
        
        return integrated_data, user_preferences
    
    except Exception as e:
        print(f"example_usage() 실행 중 오류: {e}")
        return {}, preferences


if __name__ == "__main__":
    example_usage()