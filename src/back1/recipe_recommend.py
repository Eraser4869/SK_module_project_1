# 레시피 추천 모듈
# RecipeDataIntegrator의 통합 데이터를 받아서 추천 제공

import random
from typing import Dict, List, Tuple

from recipe_data_integrator import example_usage
from crawling import food_info

class RecipeRecommender:
    """
    레시피 추천기
    RecipeDataIntegrator에서 통합된 데이터를 받아서 추천 제공
    """
    
    def __init__(self, nutrition_csv_path: str = None):
        """초기화 메서드"""
        pass
    

    def get_recommendations(self, integrated_data: Dict, 
                          recommendation_type: str = "random", 
                          top_n: int = 3) -> List[Tuple[str, Dict]]:
        """
        통합 데이터에서 추천 목록 생성
        
        Args:
            integrated_data: 통합된 레시피 데이터
            recommendation_type: 추천 방식 ("random", "diet_match", "nutrition")
            top_n: 추천할 개수
            
        Returns:
            (레시피명, 레시피데이터) 튜플 리스트
        """
        if not integrated_data:
            return []
        
        recipes = list(integrated_data.items())
        
        if recommendation_type == "random":
            random.shuffle(recipes)
            return recipes[:top_n]
        
        elif recommendation_type == "diet_match":
            def diet_score(recipe_item):
                _, recipe_data = recipe_item
                diet_classifications = recipe_data.get('diet_classifications', {})
                return sum(1 for applicable in diet_classifications.values() if applicable)
            
            recipes.sort(key=diet_score, reverse=True)
            return recipes[:top_n]
        
        elif recommendation_type == "nutrition":
            def nutrition_score(recipe_item):
                _, recipe_data = recipe_item
                nutrition_info = recipe_data.get('nutrition_info', {})
                nutrition_per_100g = nutrition_info.get('nutrition_per_100g', {})
                return nutrition_per_100g.get('protein_g', 0)
            
            recipes.sort(key=nutrition_score, reverse=True)
            return recipes[:top_n]
        
        else:
            random.shuffle(recipes)
            return recipes[:top_n]
    



    def display_recipe_summary(self, recipe_name: str, recipe_data: Dict, rank: int):
        """레시피 요약 정보 출력"""
        print(f"\n추천 레시피 #{rank}")
        print(f"메뉴명: {recipe_name}")
        
        # 식단 분류 정보
        diet_classifications = recipe_data.get('diet_classifications', {})
        applicable_diets = [diet for diet, applicable in diet_classifications.items() if applicable]
        if applicable_diets:
            print(f"식단 유형: {', '.join(applicable_diets)}")
        else:
            print("식단 유형: 일반식")
        
        # 조리 정보
        cooking_info = recipe_data.get('cooking_info', {})
        if cooking_info:
            print(f"조리시간: {cooking_info.get('predicted_time_minutes', 30):.0f}분")
            print(f"난이도: {cooking_info.get('difficulty_level', '보통')}")
        
        # 영양 정보
        nutrition_info = recipe_data.get('nutrition_info', {})
        if nutrition_info:
            nutrition_per_100g = nutrition_info.get('nutrition_per_100g', {})
            if nutrition_per_100g:
                print(f"영양정보(100g당): {nutrition_per_100g.get('kcal', 0):.0f}kcal, "
                      f"단백질 {nutrition_per_100g.get('protein_g', 0):.1f}g")
    



    # crawling
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
                print(f"  {ingredient['item']}: {ingredient['amount']}{ingredient['unit']}")
        
        # 조미료 정보
        seasonings = original_recipe.get('조미료', [])
        if seasonings:
            print(f"\n조미료:")
            for seasoning in seasonings:
                print(f"  {seasoning['item']}: {seasoning['amount']}{seasoning['unit']}")
        
        # 식단 분류 정보
        diet_classifications = recipe_data.get('diet_classifications', {})
        diet_reasons = recipe_data.get('diet_reasons', {})
        
        print(f"\n식단 분류:")
        for diet_type, applicable in diet_classifications.items():
            status = "적합" if applicable else "부적합"
            reason = diet_reasons.get(diet_type, "")
            print(f"  {diet_type}: {status}")
            if reason:
                print(f"    이유: {reason}")


        # 웹에서 가져오기        
        print("example_usage() 함수 호출 중...")
        detailed_recpie_web = food_info(recipe_name)

        print(f"상세 레시피 가져오기 완료!\n")
        print(f"{detailed_recpie_web}")

        

        """
        # 조리 정보
        cooking_info = recipe_data.get('cooking_info', {})
        if cooking_info:
            print(f"\n조리 정보:")
            print(f"  예상 조리시간: {cooking_info.get('predicted_time_minutes', 30):.0f}분")
            print(f"  난이도: {cooking_info.get('difficulty_level', '보통')}")
            print(f"  조리 종류: {cooking_info.get('cooking_kind', '알 수 없음')}")
        
        # 영양 정보
        nutrition_info = recipe_data.get('nutrition_info', {})
        if nutrition_info:
            print(f"\n영양 정보:")
            
            total_weight = nutrition_info.get('total_weight_g', 0)
            if total_weight > 0:
                print(f"  총 중량: {total_weight:.0f}g")
            
            total_nutrition = nutrition_info.get('total_nutrition', {})
            if total_nutrition:
                print(f"  총 칼로리: {total_nutrition.get('kcal', 0):.0f}kcal")
                print(f"  총 단백질: {total_nutrition.get('protein_g', 0):.1f}g")
                print(f"  총 탄수화물: {total_nutrition.get('carb_g', 0):.1f}g")
                print(f"  총 지방: {total_nutrition.get('fat_g', 0):.1f}g")
            
            nutrition_per_100g = nutrition_info.get('nutrition_per_100g', {})
            if nutrition_per_100g:
                print(f"  100g당 칼로리: {nutrition_per_100g.get('kcal', 0):.0f}kcal")
                print(f"  100g당 단백질: {nutrition_per_100g.get('protein_g', 0):.1f}g")
                print(f"  100g당 탄수화물: {nutrition_per_100g.get('carb_g', 0):.1f}g")
                print(f"  100g당 지방: {nutrition_per_100g.get('fat_g', 0):.1f}g")"""
    



    def run_recommendation_session(self, integrated_data: Dict, user_preferences: Dict = None):
        """
        통합된 데이터로 추천 세션 실행
        Args:
            integrated_data: 통합된 레시피 데이터
            user_preferences: 사용자 선호도 (호환성을 위해 유지, 사용하지 않음)
        """
        print("맞춤 레시피 추천 시스템에 오신 것을 환영합니다!")
        print("통합된 레시피 데이터 기반으로 추천을 제공합니다.")
        
        if not integrated_data:
            print("\n죄송합니다. 조건에 맞는 레시피를 찾을 수 없습니다.")
            return
        
        # 추천 생성
        recommendations = self.get_recommendations(integrated_data, "random", len(integrated_data))
        
        if not recommendations:
            print("\n죄송합니다. 조건에 맞는 레시피를 찾을 수 없습니다.")
            return
        
        print(f"\n총 {len(recommendations)}개의 레시피를 찾았습니다!")
        print("추천 점수 순으로 정렬되었습니다.")
        
        # 추천 루프 시작
        current_index = 0
        
        while current_index < len(recommendations):
            recipe_name, recipe_data = recommendations[current_index]
            
            # 레시피 요약 정보 표시
            self.display_recipe_summary(recipe_name, recipe_data, current_index + 1)
            
            # 사용자 선택 옵션 표시
            print(f"\n선택사항:")
            print("1. 이 레시피의 상세 정보 보기")
            print("2. 다른 레시피 추천받기")
            print("3. 종료")
            
            choice = input("선택하세요 (1-3): ").strip()
            
            if choice == "1":
                # 상세 레시피 표시
                self.display_detailed_recipe(recipe_name, recipe_data)
                break
                    
            elif choice == "2":
                print("\n다음 추천 레시피로 넘어갑니다.")
                current_index += 1
                
            elif choice == "3":
                print("\n추천 시스템을 종료합니다.")
                print("다음에 또 이용해주세요!")
                break
                
            else:
                print("\n잘못된 입력입니다. 다시 선택해주세요.")
            
            # 모든 레시피를 다 보여준 경우
            if current_index >= len(recommendations):
                print("\n더 이상 추천할 레시피가 없습니다.")
                print("조건을 다시 설정하거나 다음에 다시 시도해보세요.")
                break


def main():
    """메인 실행 함수"""
    try:
        print("=== 레시피 추천 시스템 시작 ===")
        print("recipe_data_integrator.py에서 통합 데이터를 가져오는 중...")
        
        # recipe_data_integrator.py의 example_usage()에서 통합 데이터 가져오기
        print("example_usage() 함수 호출 중...")
        integrated_data = example_usage()

        print(f"통합 데이터 가져오기 완료!")
        print(f"통합 데이터 타입: {type(integrated_data)}")
        
        
        if not integrated_data:
            print("오류: 통합 데이터가 비어있습니다.")
            return
        
        print(f"recipe_data_integrator.py에서 {len(integrated_data)}개 레시피의 통합 데이터를 가져왔습니다.")
        print(f"가져온 레시피 목록: {list(integrated_data.keys())}")
        
        # 첫 번째 레시피 데이터 구조 확인
        first_recipe_name = list(integrated_data.keys())[0]
        first_recipe_data = integrated_data[first_recipe_name]
        print(f"\n첫 번째 레시피 '{first_recipe_name}' 데이터 구조:")
        print(f"  - recipe_name: {first_recipe_data.get('recipe_name', 'None')}")
        print(f"  - diet_classifications: {first_recipe_data.get('diet_classifications', 'None')}")
        print(f"  - cooking_info: {first_recipe_data.get('cooking_info', 'None')}")
        print(f"  - nutrition_info 존재: {'nutrition_info' in first_recipe_data}")
        
        # 추천기 초기화
        print("\n추천기 초기화 중...")
        recommender = RecipeRecommender()
        
        print("통합 데이터로 추천 세션 실행...")
        # 통합 데이터로 추천 세션 실행
        recommender.run_recommendation_session(integrated_data)
        
    except ImportError as e:
        print(f"Import 오류: {e}")
        print("recipe_data_integrator.py 파일이 같은 디렉토리에 있는지 확인하세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()