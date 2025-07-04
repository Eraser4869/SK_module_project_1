# 레시피 데이터 통합 모듈
# 식단 분류 결과와 조리시간 예측 결과를 통합하여 완전한 레시피 정보 생성

import json
from typing import Dict, List, Any

# 기존 모듈들 import
from ai_classifier_multi_agent import MultiAgentRecipeSystem
from cooking_time_model import parse_json_and_predict


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
    
    def integrate_recipe_data(self, test_recipes: Dict, preferences: Dict = None) -> Dict:
        """
        테스트 레시피들의 식단 분류 결과와 조리시간 예측 결과를 통합
        
        Args:
            test_recipes: 분석할 레시피 딕셔너리
            preferences: 사용자 선호도 (선택적)
            
        Returns:
            통합된 레시피 데이터 딕셔너리
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
        
        # 3. 두 결과를 통합
        integrated_data = self._merge_results(test_recipes, diet_results, cooking_time_results, preferences)
        
        print(f"레시피 데이터 통합 완료!")
        return integrated_data
    
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
    
    def _merge_results(self, test_recipes: Dict, diet_results: Dict, cooking_time_results: Dict, preferences: Dict) -> Dict:
        """식단 분류 결과와 조리시간 예측 결과를 통합"""
        print("결과 통합 중...")
        
        integrated_data = {}
        
        for recipe_name, recipe_data in test_recipes.items():
            # 기본 레시피 정보
            integrated_recipe = {
                'recipe_name': recipe_name,
                'original_recipe_data': recipe_data,
                
                # 식단 분류 결과
                'diet_classifications': diet_results.get(recipe_name, {}).get('classifications', {}),
                'diet_reasons': diet_results.get(recipe_name, {}).get('reasons', {}),
                'nutrition_info': diet_results.get(recipe_name, {}).get('nutrition_info', {}),
                
                # 조리시간 예측 결과
                'cooking_info': cooking_time_results.get(recipe_name, {})
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



# 사용 예시
def example_usage():
    """통합 모듈 사용 예시"""
    
    # 하드코딩된 테스트 레시피 데이터
    test_recipes = {
        "잡채": {
            "재료": [
                {"item": "당근", "amount": 100, "unit": "g"},
                {"item": "당면", "amount": 50, "unit": "g"},
                {"item": "시금치", "amount": 100, "unit": "g"}
            ],
            "조미료": [
                {"item": "간장", "amount": 2, "unit": "큰술"},
                {"item": "설탕", "amount": 1, "unit": "큰술"}
            ]
        },
        "미역국": {
            "재료": [
                {"item": "미역", "amount": 50, "unit": "g"},
                {"item": "소고기", "amount": 100, "unit": "g"}
            ],
            "조미료": [
                {"item": "간장", "amount": 3, "unit": "큰술"},
                {"item": "참기름", "amount": 5, "unit": "작은술"}
            ]
        }
    }
    
    # 하드코딩된 테스트 선호도
    preferences = {
        "diet": [0, 0, 0, 1],    # 선호 없음
        "time": [0, 0, 0],       # 선호 없음
        "difficulty": [0, 0, 0]  # 선호 없음
    }
    
    # 통합기 초기화 및 실행
    integrator = RecipeDataIntegrator()
    result = integrator.integrate_recipe_data(test_recipes, preferences)
    
    return result




if __name__ == "__main__":
    example_usage()