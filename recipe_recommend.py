# 레시피 추천 모듈
# RecipeDataIntegrator의 통합 데이터를 받아서 선호도 기반 추천 제공

import random
from typing import Dict, List, Tuple

from recipe_data_integrator import example_usage
from crawling import food_info

class RecipeRecommender:
    """
    레시피 추천기
    RecipeDataIntegrator에서 통합된 데이터를 받아서 사용자 선호도 기반 추천 제공
    """
    
    def __init__(self, nutrition_csv_path: str = None):
        """초기화 메서드"""
        self.preference_mapping = {
            'diet': ['다이어트', '저탄고지', '저염', '채식'],
            'time': ['15분', '30분', '45분'],
            'difficulty': ['쉬움', '보통', '어려움']
        }
    

    def get_recommendations_by_preference(self, integrated_data: Dict, user_preferences: Dict, top_n: int = 3) -> List[Tuple[str, Dict]]:
        """
        사용자 선호도 기반으로 추천 목록 생성
        
        Args:
            integrated_data: 통합된 레시피 데이터
            user_preferences: 사용자 선호도
            top_n: 추천할 개수
            
        Returns:
            (레시피명, 레시피데이터) 튜플 리스트 (선호도 점수 순으로 정렬)
        """
        if not integrated_data:
            return []
        
        recipes = list(integrated_data.items())
        
        # 선호도 점수 기준으로 정렬 (높은 점수가 먼저)
        def preference_score(recipe_item):
            _, recipe_data = recipe_item
            return recipe_data.get('preference_score', 0)
        
        recipes.sort(key=preference_score, reverse=True)
        return recipes[:top_n]


    def display_user_preferences(self, user_preferences: Dict):
        """사용자 선호도 정보 출력"""
        print(f"\n사용자 선호도 설정:")
        print("="*40)
        
        # 식단 선호도
        diet_prefs = user_preferences.get('diet', [0, 0, 0, 0])
        preferred_diets = [self.preference_mapping['diet'][i] for i, pref in enumerate(diet_prefs) if pref == 1]
        if preferred_diets:
            print(f"선호 식단: {', '.join(preferred_diets)}")
        else:
            print(f"선호 식단: 없음")
        
        # 시간 선호도
        time_prefs = user_preferences.get('time', [0, 0, 0])
        preferred_times = [self.preference_mapping['time'][i] for i, pref in enumerate(time_prefs) if pref == 1]
        if preferred_times:
            print(f"선호 조리시간: {', '.join(preferred_times)}")
        else:
            print(f"선호 조리시간: 없음")
        
        # 난이도 선호도
        difficulty_prefs = user_preferences.get('difficulty', [0, 0, 0])
        preferred_difficulties = [self.preference_mapping['difficulty'][i] for i, pref in enumerate(difficulty_prefs) if pref == 1]
        if preferred_difficulties:
            print(f"선호 난이도: {', '.join(preferred_difficulties)}")
        else:
            print(f"선호 난이도: 없음")


    def display_recipe_summary(self, recipe_name: str, recipe_data: Dict, rank: int = 1):
        """레시피 요약 정보 출력 (선호도 매칭 정보 포함)"""
        preference_score = recipe_data.get('preference_score', 0)
        
        print(f"\n추천 레시피 #{rank}: {recipe_name} (선호도 점수: {preference_score:.0f}/100)")
        print("="*60)
        
        # 선호도 매칭 정보
        preference_matches = recipe_data.get('preference_matches', {})
        
        # 식단 매칭
        diet_matches = preference_matches.get('diet_matches', [])
        if diet_matches:
            print(f"식단 매칭: {', '.join(diet_matches)}")
        else:
            print(f"식단 매칭: 없음")
        
        # 시간 매칭
        time_match = preference_matches.get('time_match', False)
        cooking_info = recipe_data.get('cooking_info', {})
        predicted_time = cooking_info.get('predicted_time_minutes', 30)
        if time_match:
            print(f"시간 매칭: {predicted_time:.0f}분 (선호 시간대와 일치)")
        else:
            print(f"시간 매칭: {predicted_time:.0f}분 (선호 시간대와 불일치)")
        
        # 난이도 매칭
        difficulty_match = preference_matches.get('difficulty_match', False)
        difficulty_level = cooking_info.get('difficulty_level', '보통')
        if difficulty_match:
            print(f"난이도 매칭: {difficulty_level} (선호 난이도와 일치)")
        else:
            print(f"난이도 매칭: {difficulty_level} (선호 난이도와 불일치)")
        
        # 식단 분류 정보
        diet_classifications = recipe_data.get('diet_classifications', {})
        applicable_diets = [diet for diet, applicable in diet_classifications.items() if applicable]
        if applicable_diets:
            print(f"식단 유형: {', '.join(applicable_diets)}")
        else:
            print("식단 유형: 일반식")
        
        # 영양 정보
        nutrition_info = recipe_data.get('nutrition_info', {})
        if nutrition_info:
            nutrition_per_100g = nutrition_info.get('nutrition_per_100g', {})
            if nutrition_per_100g:
                print(f"영양정보(100g당): {nutrition_per_100g.get('kcal', 0):.0f}kcal, "
                      f"단백질 {nutrition_per_100g.get('protein_g', 0):.1f}g")


    def display_detailed_recipe(self, recipe_name: str, recipe_data: Dict):
        """상세 레시피 정보 출력"""
        print(f"\n상세 정보: {recipe_name}")
        print("="*60)
        
        # 원본 레시피 정보
        original_recipe = recipe_data.get('original_recipe_data', {})
        
        # 재료 정보
        ingredients = original_recipe.get('재료', [])
        if ingredients:
            print(f"\n재료:")
            for ingredient in ingredients:
                print(f"  • {ingredient['item']}: {ingredient['amount']}{ingredient['unit']}")
        
        # 조미료 정보
        seasonings = original_recipe.get('조미료', [])
        if seasonings:
            print(f"\n조미료:")
            for seasoning in seasonings:
                print(f"  • {seasoning['item']}: {seasoning['amount']}{seasoning['unit']}")
        
        # 식단 분류 상세 정보
        diet_classifications = recipe_data.get('diet_classifications', {})
        diet_reasons = recipe_data.get('diet_reasons', {})
        
        print(f"\n 식단 분류 상세:")
        for diet_type, applicable in diet_classifications.items():
            status = "적합 O" if applicable else "부적합 X"
            reason = diet_reasons.get(diet_type, "")
            print(f"  {diet_type}: {status}")
            if reason:
                print(f"이유: {reason}")

        # 웹에서 상세 레시피 가져오기        
        print(f"\n웹에서 상세 레시피 가져오는 중...")
        try:
            detailed_recipe_web = food_info(recipe_name)
            
            if detailed_recipe_web:
                print(f"\n상세 레시피:")
                print(f"재료 목록:\n{detailed_recipe_web.get('ingredients', '정보 없음')}")
                print(f"\n조리 방법:")
                for step in detailed_recipe_web.get('recipe', []):
                    print(f"  {step}")
            else:
                print(f" 웹에서 '{recipe_name}' 레시피를 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"웹 레시피 가져오기 실패: {e}")


    def run_preference_based_recommendation(self, integrated_data: Dict, user_preferences: Dict):
        """
        사용자 선호도 기반 추천 세션 실행
        Args:
            integrated_data: 통합된 레시피 데이터
            user_preferences: 사용자 선호도
        """
        print("맞춤 레시피 추천 시스템에 오신 것을 환영합니다!")
        print("사용자 선호도를 기반으로 최적의 레시피를 추천해드립니다.")
        
        # 사용자 선호도 표시
        self.display_user_preferences(user_preferences)
        
        if not integrated_data:
            print("\n죄송합니다. 조건에 맞는 레시피를 찾을 수 없습니다.")
            return
        
        # 선호도 기반 추천 생성
        recommendations = self.get_recommendations_by_preference(integrated_data, user_preferences, len(integrated_data))
        
        if not recommendations:
            print("\n죄송합니다. 조건에 맞는 레시피를 찾을 수 없습니다.")
            return
        
        print(f"\n총 {len(recommendations)}개의 레시피를 분석했습니다!")
        
        # 가장 높은 점수의 레시피 선택 (첫 번째)
        best_recipe_name, best_recipe_data = recommendations[0]
        best_score = best_recipe_data.get('preference_score', 0)
        
        print(f"선호도 점수가 가장 높은 레시피를 추천드립니다!")
        
        # 레시피 요약 정보 표시
        self.display_recipe_summary(best_recipe_name, best_recipe_data, 1)
        
        # 다른 추천 레시피들도 간단히 표시
        if len(recommendations) > 1:
            print(f"\n다른 추천 레시피들:")
            for i, (recipe_name, recipe_data) in enumerate(recommendations[1:], 2):
                score = recipe_data.get('preference_score', 0)
                print(f"  #{i}. {recipe_name} (점수: {score:.0f}/100)")
        
        # 바로 상세 레시피 표시
        self.display_detailed_recipe(best_recipe_name, best_recipe_data)
        
        print(f"\n'{best_recipe_name}' 레시피 추천이 완료되었습니다!")
        print(f"선호도 매칭률: {best_score:.0f}/100점")
        print("맛있는 요리 되세요!")


def main():
    """메인 실행 함수"""
    try:
        print("=== 선호도 기반 레시피 추천 시스템 시작 ===")
        print("recipe_data_integrator.py에서 통합 데이터를 가져오는 중...")
        
        # recipe_data_integrator.py의 example_usage()에서 통합 데이터와 선호도 가져오기
        print("example_usage() 함수 호출 중...")
        
        try:
            result = example_usage()
            
            # 반환값 타입 확인 및 디버깅
            print(f"example_usage() 반환값 타입: {type(result)}")
            
            # 튜플이 반환되었는지 확인
            if isinstance(result, tuple) and len(result) == 2:
                integrated_data, user_preferences = result
                print(f"튜플 형태로 데이터를 성공적으로 받았습니다.")
            else:
                print("오류: example_usage()가 튜플을 반환하지 않았습니다.")
                print("단일 값으로 처리를 시도합니다...")
                integrated_data = result
                user_preferences = {
                    "diet": [0, 0, 0, 1],    # 채식 선호
                    "time": [1, 0, 0],       # 15분 이내 선호
                    "difficulty": [1, 0, 0]  # 쉬운 난이도 선호
                }
            
            print(f"integrated_data 타입: {type(integrated_data)}")
            
            if not integrated_data:
                print("오류: 통합 데이터가 비어있습니다.")
                return
            
            if not isinstance(integrated_data, dict):
                print(f"오류: integrated_data가 딕셔너리가 아닙니다. 타입: {type(integrated_data)}")
                return
        
        except Exception as e:
            print(f"example_usage() 호출 실패: {e}")
            print("백업 방법으로 직접 데이터를 생성합니다...")
            return
        
        print(f"{len(integrated_data)}개 레시피의 통합 데이터를 성공적으로 가져왔습니다.")
        print(f"분석된 레시피 목록: {list(integrated_data.keys())}")
        
        # 선호도 점수 확인
        scores = [recipe_data.get('preference_score', 0) for recipe_data in integrated_data.values()]
        print(f"선호도 점수 범위: {min(scores):.0f} ~ {max(scores):.0f}점")
        
        # 추천기 초기화
        print("\n추천기 초기화 중...")
        recommender = RecipeRecommender()
        
        print("선호도 기반 맞춤 추천을 시작합니다...\n")
        # 선호도 기반 추천 세션 실행
        recommender.run_preference_based_recommendation(integrated_data, user_preferences)
        
    except ImportError as e:
        print(f"Import 오류: {e}")
        print("recipe_data_integrator.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()