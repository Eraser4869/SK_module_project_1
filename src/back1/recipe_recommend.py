# AI 기반 레시피 추천 시스템
# 기존 분류기 + 조리시간 + 사용자 맞춤 추천
# 국가표준식품성분표 사용 버전

import pandas as pd
import numpy as np
import json
import os
import requests
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from ai_classifier import SmartRecipeClassifier  # 수정된 분류기 import

class RecipeRecommendationSystem:
    # AI 기반 레시피 추천 시스템
    # 국가표준식품성분표를 기반으로 한 정확한 영양소 분석과 맞춤 추천 제공
    
    def __init__(self, nutrition_csv_path: str, recipe_database_path: str = None):
        # 시스템 초기화 메서드
        # nutrition_csv_path: 국가표준식품성분표 CSV 파일 경로
        # recipe_database_path: 레시피 데이터베이스 경로 (선택적)
        
        # 기존 분류기 초기화 (국가표준식품성분표 사용)
        self.classifier = SmartRecipeClassifier(nutrition_csv_path, use_ai=True)
        
        # 레시피 데이터베이스 로드
        self.recipe_database = self._load_recipe_database(recipe_database_path)
        
        # 추천 히스토리 (사용자별 기록 저장)
        self.recommendation_history = {}
        
        # 현재 추천 세션 상태
        self.current_session = {
            'user_preferences': {},     # 사용자 선호도
            'recommended_recipes': [],  # 추천된 레시피 목록
            'current_index': 0         # 현재 보고 있는 레시피 인덱스
        }
        
        print("레시피 추천 시스템이 준비되었습니다!")
        print("국가표준식품성분표 기반으로 정확한 영양소 분석을 제공합니다.")
    
    def _load_recipe_database(self, database_path: str) -> List[Dict]:
        # 레시피 데이터베이스 로드 메서드
        # 실제 환경에서는 외부 API나 DB에서 가져올 예정
        # 현재는 샘플 데이터 사용
        # 반환값: 레시피 딕셔너리 리스트
        
        # 임시 샘플 레시피 데이터베이스
        # 국가표준식품성분표에 있는 재료명으로 구성
        sample_recipes = [
            {
                "name": "닭가슴살 샐러드",
                "cooking_time": 15,
                "difficulty": "쉬움",
                "ingredients": {
                    "재료": [
                        {"item": "닭고기", "amount": 150, "unit": "g"},        # 국가표준: 닭고기로 매칭
                        {"item": "상추", "amount": 100, "unit": "g"},         # 국가표준: 상추로 매칭
                        {"item": "토마토", "amount": 10, "unit": "개"},        # 국가표준: 토마토로 매칭
                        {"item": "오이", "amount": 50, "unit": "g"}           # 국가표준: 오이로 매칭
                    ],
                    "조미료": [
                        {"item": "올리브유", "amount": 1, "unit": "큰술"},     # 국가표준: 올리브유로 매칭
                        {"item": "레몬", "amount": 1, "unit": "큰술"},         # 국가표준: 레몬으로 매칭 예상
                        {"item": "소금", "amount": 1, "unit": "조금"}          # 국가표준: 소금으로 매칭
                    ]
                },
                "description": "건강하고 맛있는 다이어트 샐러드",
                "detailed_recipe": [
                    "1. 닭가슴살을 소금, 후추로 밑간하고 팬에 구워줍니다.",
                    "2. 상추와 채소들을 깨끗이 씻어 적당한 크기로 자릅니다.",
                    "3. 올리브유와 레몬즙을 섞어 드레싱을 만듭니다.",
                    "4. 모든 재료를 접시에 담고 드레싱을 뿌려 완성합니다."
                ]
            },
            {
                "name": "연어 아보카도 샐러드", 
                "cooking_time": 20,
                "difficulty": "쉬움",
                "ingredients": {
                    "재료": [
                        {"item": "연어", "amount": 200, "unit": "g"},         # 국가표준: 연어로 매칭
                        {"item": "아보카도", "amount": 1, "unit": "개"},       # 국가표준에서 찾기 어려울 수 있음
                        {"item": "시금치", "amount": 100, "unit": "g"},       # 국가표준: 시금치로 매칭
                        {"item": "견과류", "amount": 30, "unit": "g"}         # 국가표준: 견과류 카테고리에서 매칭
                    ],
                    "조미료": [
                        {"item": "올리브유", "amount": 2, "unit": "큰술"},
                        {"item": "레몬", "amount": 1, "unit": "큰술"},
                        {"item": "소금", "amount": 1, "unit": "조금"}
                    ]
                },
                "description": "오메가3가 풍부한 건강 샐러드",
                "detailed_recipe": [
                    "1. 연어를 구워서 한입 크기로 자릅니다.",
                    "2. 아보카도를 슬라이스합니다.",
                    "3. 시금치를 깨끗이 씻어 준비합니다.",
                    "4. 드레싱을 만들어 모든 재료와 함께 버무립니다."
                ]
            },
            {
                "name": "두부 김치찌개",
                "cooking_time": 30,
                "difficulty": "보통",
                "ingredients": {
                    "재료": [
                        {"item": "배추김치", "amount": 200, "unit": "g"},     # 국가표준: 배추김치로 매칭
                        {"item": "돼지고기", "amount": 150, "unit": "g"},     # 국가표준: 돼지고기로 매칭
                        {"item": "두부", "amount": 1, "unit": "모"},          # 국가표준: 두부로 매칭
                        {"item": "파", "amount": 2, "unit": "줄기"}           # 국가표준: 파로 매칭
                    ],
                    "조미료": [
                        {"item": "간장", "amount": 2, "unit": "큰술"},        # 국가표준: 간장으로 매칭
                        {"item": "고춧가루", "amount": 1, "unit": "큰술"},    # 국가표준: 고춧가루로 매칭 (fallback에 있음)
                        {"item": "마늘", "amount": 2, "unit": "쪽"},          # 국가표준: 마늘로 매칭
                        {"item": "물", "amount": 500, "unit": "ml"}           # fallback: 물로 매칭
                    ]
                },
                "description": "한국인의 소울푸드 김치찌개",
                "detailed_recipe": [
                    "1. 돼지고기를 먼저 볶아줍니다.",
                    "2. 김치를 넣고 함께 볶습니다.",
                    "3. 물을 넣고 끓인 후 두부를 넣습니다.",
                    "4. 간을 맞추고 대파를 넣어 마무리합니다."
                ]
            },
            {
                "name": "브로콜리 현미밥",
                "cooking_time": 25,
                "difficulty": "쉬움",
                "ingredients": {
                    "재료": [
                        {"item": "현미", "amount": 100, "unit": "g"},         # 국가표준: 현미로 매칭
                        {"item": "브로콜리", "amount": 200, "unit": "g"},     # 국가표준: 브로콜리로 매칭
                        {"item": "토마토", "amount": 100, "unit": "g"},       # 국가표준: 토마토로 매칭
                        {"item": "양파", "amount": 1, "unit": "개"}           # 국가표준: 양파로 매칭
                    ],
                    "조미료": [
                        {"item": "올리브유", "amount": 2, "unit": "큰술"},
                        {"item": "소금", "amount": 1, "unit": "조금"},
                        {"item": "후추", "amount": 1, "unit": "조금"}
                    ]
                },
                "description": "영양 가득한 채식 건강식",
                "detailed_recipe": [
                    "1. 현미를 물에 삶아 준비합니다.",
                    "2. 브로콜리를 살짝 데쳐줍니다.",
                    "3. 토마토와 양파를 자릅니다.",
                    "4. 모든 재료를 볼에 담고 조미료와 함께 섞습니다."
                ]
            },
            {
                "name": "소고기 스테이크",
                "cooking_time": 40,
                "difficulty": "어려움",
                "ingredients": {
                    "재료": [
                        {"item": "쇠고기", "amount": 300, "unit": "g"},       # 국가표준: 쇠고기로 매칭
                        {"item": "버터", "amount": 50, "unit": "g"},          # 국가표준: 버터로 매칭
                        {"item": "마늘", "amount": 3, "unit": "쪽"},          # 국가표준: 마늘로 매칭
                        {"item": "양파", "amount": 1, "unit": "개"}           # 국가표준: 양파로 매칭
                    ],
                    "조미료": [
                        {"item": "소금", "amount": 1, "unit": "큰술"},
                        {"item": "후추", "amount": 1, "unit": "작은술"},
                        {"item": "올리브유", "amount": 2, "unit": "큰술"}
                    ]
                },
                "description": "고급 레스토랑 스타일 스테이크",
                "detailed_recipe": [
                    "1. 고기를 실온에 30분간 두어 온도를 맞춥니다.",
                    "2. 소금과 후추로 시즈닝합니다.",
                    "3. 팬을 충분히 달군 후 고기를 굽습니다.",
                    "4. 버터와 마늘로 풍미를 더해 완성합니다."
                ]
            }
        ]
        
        print("샘플 레시피 데이터를 국가표준식품성분표로 분석 중...")
        
        # 각 레시피를 분류하여 태그 추가
        for i, recipe in enumerate(sample_recipes, 1):
            print(f"레시피 {i}/{len(sample_recipes)} 분석 중: {recipe['name']}")
            
            # 분류기를 사용하여 레시피 분석
            classification_result = self.classifier.classify_recipe(
                recipe["name"], 
                recipe["ingredients"]
            )
            
            # 분류 결과를 레시피에 추가
            recipe["diet_tags"] = {
                "다이어트": classification_result["classifications"]["다이어트"],
                "저탄고지": classification_result["classifications"]["저탄고지"], 
                "저염": classification_result["classifications"]["저염"],
                "채식": classification_result["classifications"]["채식"]
            }
            
            # 영양소 정보 추가 (국가표준식품성분표 컬럼명 사용)
            recipe["nutrition_per_100g"] = classification_result["nutrition_result"]["nutrition_per_100g"]
            
        print("레시피 분석 완료!")
        return sample_recipes
    
    def get_user_preferences(self) -> Dict:
        # 사용자 선호도 입력받기 메서드
        # 대화형 인터페이스로 사용자의 식단 선호도, 조리시간, 난이도 등을 수집
        # 반환값: 사용자 선호도 딕셔너리
        
        print("\n=== 맞춤 레시피 추천을 위한 정보 입력 ===")
        print("국가표준식품성분표 기반으로 정확한 영양소 분석을 제공합니다.")
        
        preferences = {}
        
        # 식단 유형 선호도 입력
        print("\n원하는 식단 유형을 선택하세요 (여러 개 선택 가능):")
        diet_types = ["다이어트", "저탄고지", "저염", "채식"]
        
        for i, diet_type in enumerate(diet_types, 1):
            print(f"{i}. {diet_type}")
        
        selected = input("번호를 입력하세요 (예: 1,3): ").strip()
        preferences["preferred_diets"] = []
        
        if selected:
            try:
                # 쉼표로 구분된 번호들을 파싱
                indices = [int(x.strip()) - 1 for x in selected.split(",")]
                preferences["preferred_diets"] = [diet_types[i] for i in indices if 0 <= i < len(diet_types)]
                print(f"선택된 식단: {', '.join(preferences['preferred_diets'])}")
            except:
                print("잘못된 입력입니다. 모든 식단을 고려합니다.")
        
        # 조리시간 선호도 입력
        print("\n희망 조리시간을 선택하세요:")
        print("1. 15분 이하 (간단)")
        print("2. 30분 이하 (보통)")
        print("3. 45분 이하 (정성)")
        print("4. 상관없음")
        
        time_choice = input("번호를 입력하세요: ").strip()
        time_limits = {
            "1": 15,
            "2": 30, 
            "3": 45,
            "4": 999
        }
        preferences["max_cooking_time"] = time_limits.get(time_choice, 999)
        
        # 난이도 선호도 입력
        print("\n희망 난이도를 선택하세요:")
        print("1. 쉬움")
        print("2. 보통")
        print("3. 어려움") 
        print("4. 상관없음")
        
        difficulty_choice = input("번호를 입력하세요: ").strip()
        difficulty_map = {
            "1": ["쉬움"],
            "2": ["쉬움", "보통"],
            "3": ["쉬움", "보통", "어려움"],
            "4": ["쉬움", "보통", "어려움"]
        }
        preferences["allowed_difficulty"] = difficulty_map.get(difficulty_choice, ["쉬움", "보통", "어려움"])
        
        # 현재 세션에 사용자 선호도 저장
        self.current_session["user_preferences"] = preferences
        return preferences
    
    def calculate_recipe_score(self, recipe: Dict, preferences: Dict) -> float:
        # 레시피와 사용자 선호도의 매칭 점수 계산 메서드
        # 다양한 요소를 고려한 가중치 기반 점수 계산
        # recipe: 레시피 정보 딕셔너리
        # preferences: 사용자 선호도 딕셔너리
        # 반환값: 0.0 ~ 1.0 사이의 점수
        
        score = 0.0
        
        # 식단 타입 매칭 (가중치: 40%) - 가장 중요한 요소
        if preferences.get("preferred_diets"):
            diet_matches = 0
            for diet in preferences["preferred_diets"]:
                if recipe["diet_tags"].get(diet, False):
                    diet_matches += 1
            
            if diet_matches > 0:
                # 매칭된 식단 비율에 따라 점수 부여
                score += 0.4 * (diet_matches / len(preferences["preferred_diets"]))
        else:
            score += 0.2  # 선호도 없으면 기본점수
        
        # 조리시간 매칭 (가중치: 30%)
        max_time = preferences.get("max_cooking_time", 999)
        if recipe["cooking_time"] <= max_time:
            # 짧을수록 높은 점수 (시간 효율성 고려)
            time_score = 1.0 - (recipe["cooking_time"] / max_time) * 0.5
            score += 0.3 * time_score
        
        # 난이도 매칭 (가중치: 20%)
        allowed_difficulty = preferences.get("allowed_difficulty", ["쉬움", "보통", "어려움"])
        if recipe["difficulty"] in allowed_difficulty:
            score += 0.2
        
        # 영양 균형 (가중치: 10%) - 국가표준식품성분표 데이터 활용
        nutrition = recipe["nutrition_per_100g"]
        
        # 단백질 비율 평가 (20g을 기준으로 정규화)
        protein_ratio = min(nutrition.get("protein_g", 0) / 20, 1.0)
        
        # 칼로리 적정성 평가 (300kcal을 기준으로)
        calorie_appropriateness = 1.0 - abs(nutrition.get("kcal", 0) - 200) / 300
        calorie_appropriateness = max(0, calorie_appropriateness)
        
        # 영양 점수 = (단백질 점수 + 칼로리 적정성) / 2
        nutrition_score = (protein_ratio + calorie_appropriateness) / 2
        score += 0.1 * nutrition_score
        
        return min(score, 1.0)  # 최대 1.0점으로 제한
    
    def get_recommended_recipes(self, preferences: Dict) -> List[Tuple[Dict, float]]:
        # 사용자 선호도에 따른 레시피 추천 목록 생성 메서드
        # preferences: 사용자 선호도 딕셔너리
        # 반환값: (레시피, 점수) 튜플의 리스트 (점수 순으로 정렬됨)
        
        scored_recipes = []
        
        # 모든 레시피에 대해 점수 계산
        for recipe in self.recipe_database:
            score = self.calculate_recipe_score(recipe, preferences)
            scored_recipes.append((recipe, score))
        
        # 점수 순으로 내림차순 정렬 (높은 점수가 먼저)
        scored_recipes.sort(key=lambda x: x[1], reverse=True)
        
        return scored_recipes
    
    def display_recipe_summary(self, recipe: Dict, score: float, rank: int):
        # 레시피 요약 정보 출력 메서드
        # recipe: 레시피 정보 딕셔너리
        # score: 추천 점수
        # rank: 추천 순위
        
        print(f"\n=== 추천 레시피 #{rank} ===")
        print(f"메뉴명: {recipe['name']}")
        print(f"조리시간: {recipe['cooking_time']}분")
        print(f"난이도: {recipe['difficulty']}")
        print(f"추천점수: {score:.1%}")
        
        # 해당하는 식단 태그 표시
        applicable_tags = [tag for tag, applicable in recipe["diet_tags"].items() if applicable]
        if applicable_tags:
            print(f"식단 유형: {', '.join(applicable_tags)}")
        else:
            print("식단 유형: 일반식")
        
        print(f"설명: {recipe['description']}")
        
        # 주요 영양소 정보 (국가표준식품성분표 컬럼명 사용)
        nutrition = recipe["nutrition_per_100g"]
        print(f"영양정보 (100g당): 칼로리 {nutrition.get('kcal', 0):.0f}kcal, "
              f"단백질 {nutrition.get('protein_g', 0):.1f}g, "
              f"탄수화물 {nutrition.get('carb_g', 0):.1f}g, "
              f"당류 {nutrition.get('당류', 0):.1f}g")  # 국가표준 컬럼명
    
    def get_recipe_image_url(self, recipe_name: str) -> str:
        # 레시피 이미지 URL 생성 메서드
        # 실제 환경에서는 AI 이미지 생성(DALL-E, Midjourney) 또는 웹 스크래핑 사용
        # recipe_name: 레시피 이름
        # 반환값: 이미지 URL 문자열
        
        # 현재는 플레이스홀더 이미지 사용
        # 실제 구현시에는 OpenAI DALL-E API, Midjourney API 등을 사용할 예정
        
        placeholder_images = {
            "닭가슴살 샐러드": "https://via.placeholder.com/400x300/87CEEB/000000?text=Chicken+Salad",
            "연어 아보카도 샐러드": "https://via.placeholder.com/400x300/98FB98/000000?text=Salmon+Avocado+Salad",
            "두부 김치찌개": "https://via.placeholder.com/400x300/FF6347/000000?text=Kimchi+Stew",
            "브로콜리 현미밥": "https://via.placeholder.com/400x300/90EE90/000000?text=Healthy+Rice+Bowl",
            "소고기 스테이크": "https://via.placeholder.com/400x300/8B4513/000000?text=Beef+Steak"
        }
        
        return placeholder_images.get(recipe_name, "https://via.placeholder.com/400x300/CCCCCC/000000?text=Recipe+Image")
    
    def display_detailed_recipe(self, recipe: Dict):
        # 상세 레시피 정보 출력 메서드
        # recipe: 레시피 정보 딕셔너리
        
        print(f"\n{'='*60}")
        print(f"상세 레시피: {recipe['name']}")
        print(f"{'='*60}")
        
        print(f"\n조리시간: {recipe['cooking_time']}분")
        print(f"난이도: {recipe['difficulty']}")
        
        # 재료 목록 출력
        print(f"\n재료:")
        for ingredient in recipe["ingredients"]["재료"]:
            print(f"  - {ingredient['item']}: {ingredient['amount']}{ingredient['unit']}")
        
        # 조미료 목록 출력
        print(f"\n조미료:")
        for seasoning in recipe["ingredients"]["조미료"]:
            print(f"  - {seasoning['item']}: {seasoning['amount']}{seasoning['unit']}")
        
        # 조리법 단계별 출력
        print(f"\n조리법:")
        for i, step in enumerate(recipe["detailed_recipe"], 1):
            print(f"  {i}. {step}")
        
        # 상세 영양소 정보 출력 (국가표준식품성분표 기반)
        nutrition = recipe["nutrition_per_100g"]
        print(f"\n상세 영양소 정보 (100g당):")
        print(f"  • 칼로리: {nutrition.get('kcal', 0):.1f} kcal")
        print(f"  • 탄수화물: {nutrition.get('carb_g', 0):.1f}g")
        print(f"  • 단백질: {nutrition.get('protein_g', 0):.1f}g")
        print(f"  • 지방: {nutrition.get('fat_g', 0):.1f}g")
        print(f"  • 나트륨: {nutrition.get('sodium_mg', 0):.1f}mg")
        print(f"  • 당류: {nutrition.get('당류', 0):.1f}g")  # 국가표준 컬럼명
        print(f"  • 식이섬유: {nutrition.get('총  식이섬유', 0):.1f}g")  # 국가표준 컬럼명 (공백 포함)
        
        # 식단 분류 정보 출력
        applicable_tags = [tag for tag, applicable in recipe["diet_tags"].items() if applicable]
        if applicable_tags:
            print(f"\n적합한 식단: {', '.join(applicable_tags)}")
        else:
            print(f"\n적합한 식단: 일반식")
        
        # 이미지 URL 표시
        image_url = self.get_recipe_image_url(recipe['name'])
        print(f"\n레시피 이미지: {image_url}")
        
        print(f"\n{'='*60}")
    
    def run_recommendation_session(self):
        # 추천 세션 실행 메서드
        # 사용자와의 대화형 인터페이스를 통해 전체 추천 프로세스 진행
        
        print("맞춤 레시피 추천 시스템에 오신 것을 환영합니다!")
        print("국가표준식품성분표 기반으로 정확한 영양소 분석을 제공합니다.")
        
        # 사용자 선호도 입력받기
        preferences = self.get_user_preferences()
        
        # 추천 레시피 생성
        print("\n선호도에 맞는 레시피를 검색 중...")
        recommended_recipes = self.get_recommended_recipes(preferences)
        self.current_session["recommended_recipes"] = recommended_recipes
        
        if not recommended_recipes:
            print("\n죄송합니다. 조건에 맞는 레시피를 찾을 수 없습니다.")
            print("조건을 다시 설정하거나 다른 선호도를 시도해보세요.")
            return
        
        print(f"\n총 {len(recommended_recipes)}개의 레시피를 찾았습니다!")
        print("추천 점수 순으로 정렬되었습니다.")
        
        # 추천 루프 시작
        current_index = 0
        
        while current_index < len(recommended_recipes):
            recipe, score = recommended_recipes[current_index]
            
            # 레시피 요약 정보 표시
            self.display_recipe_summary(recipe, score, current_index + 1)
            
            # 이미지 URL 표시
            image_url = self.get_recipe_image_url(recipe['name'])
            print(f"이미지: {image_url}")
            
            # 사용자 선택 옵션 표시
            print(f"\n선택사항:")
            print("1. 이 레시피의 상세 정보 보기")
            print("2. 다른 레시피 추천받기")
            print("3. 종료")
            
            choice = input("선택하세요 (1-3): ").strip()
            
            if choice == "1":
                # 상세 레시피 표시
                self.display_detailed_recipe(recipe)
                
                # 만족도 확인
                print("\n이 레시피가 마음에 드시나요?")
                satisfied = input("만족하시면 'y' 또는 '네'를 입력하세요: ").strip().lower()
                
                if satisfied in ['y', 'yes', '네', '좋아요', '만족', 'ㅇ']:
                    print("\n추천이 도움이 되어 기쁩니다! 맛있게 드세요!")
                    print("이 레시피는 국가표준식품성분표를 기반으로 정확한 영양소 분석을 제공했습니다.")
                    break
                else:
                    print("\n다른 레시피를 찾아보겠습니다.")
                    current_index += 1
                    
            elif choice == "2":
                print("\n⏭다음 추천 레시피로 넘어갑니다.")
                current_index += 1
                
            elif choice == "3":
                print("\n추천 시스템을 종료합니다.")
                print("다음에 또 이용해주세요!")
                break
                
            else:
                print("\n잘못된 입력입니다. 다시 선택해주세요.")
            
            # 모든 레시피를 다 보여준 경우
            if current_index >= len(recommended_recipes):
                print("\n더 이상 추천할 레시피가 없습니다.")
                print("조건을 다시 설정하거나 다음에 다시 시도해보세요.")
                
                # 다시 시작할지 물어보기
                restart = input("\n다시 추천받으시겠습니까? (y/n): ").strip().lower()
                if restart in ['y', 'yes', '네', 'ㅇ']:
                    self.run_recommendation_session()  # 재귀 호출로 다시 시작
                break
    
    def get_cooking_time_estimate(self, recipe_data: Dict) -> int:
        # 조리시간 추정 메서드 (다른 팀의 조리시간 분석 모듈 연동 예정)
        # 현재는 간단한 휴리스틱 사용, 추후 AI 모델로 업그레이드 예정
        # recipe_data: 레시피 재료 정보 딕셔너리
        # 반환값: 예상 조리시간 (분 단위)
        
        # 재료 개수에 따른 기본 시간 계산
        ingredient_count = len(recipe_data.get("재료", [])) + len(recipe_data.get("조미료", []))
        
        # 재료 개수당 2분씩 기본 시간 할당
        base_time = ingredient_count * 2
        
        # 복잡한 조리가 필요한 재료 확인
        complex_ingredients = ["고기", "생선", "해산물", "닭", "소", "돼지", "연어"]
        meat_cooking_time = 0
        
        total_weight = 0
        for ingredient in recipe_data.get("재료", []):
            item_name = ingredient.get("item", "").lower()
            amount = ingredient.get("amount", 0)
            
            total_weight += amount
            
            # 복잡한 재료 확인 (고기류는 추가 조리시간 필요)
            if any(complex_ing in item_name for complex_ing in complex_ingredients):
                meat_cooking_time += 15  # 고기류 조리시간 추가
        
        # 총 중량에 따른 시간 조정
        weight_time = 0
        if total_weight > 500:  # 500g 이상
            weight_time = 20
        elif total_weight > 300:  # 300g 이상
            weight_time = 10
        elif total_weight > 150:  # 150g 이상
            weight_time = 5
        
        # 총 조리시간 = 기본시간 + 고기조리시간 + 중량조정시간
        total_time = base_time + meat_cooking_time + weight_time
        
        # 최소 10분, 최대 120분으로 제한
        return min(max(total_time, 10), 120)
    
    def analyze_recipe_trends(self) -> Dict:
        # 레시피 트렌드 분석 메서드
        # 현재 데이터베이스의 레시피들을 분석하여 트렌드 정보 제공
        # 반환값: 트렌드 분석 결과 딕셔너리
        
        trends = {
            "diet_popularity": {},      # 식단별 인기도
            "avg_cooking_time": 0,      # 평균 조리시간
            "difficulty_distribution": {},  # 난이도 분포
            "nutrition_averages": {}    # 평균 영양소
        }
        
        # 식단별 인기도 계산
        diet_counts = {"다이어트": 0, "저탄고지": 0, "저염": 0, "채식": 0}
        total_cooking_time = 0
        difficulty_counts = {"쉬움": 0, "보통": 0, "어려움": 0}
        
        nutrition_totals = {
            "kcal": 0, "protein_g": 0, "carb_g": 0, "fat_g": 0,
            "당류": 0, "총  식이섬유": 0, "sodium_mg": 0
        }
        
        for recipe in self.recipe_database:
            # 식단 태그 카운트
            for diet_type, is_applicable in recipe["diet_tags"].items():
                if is_applicable:
                    diet_counts[diet_type] += 1
            
            # 조리시간 누적
            total_cooking_time += recipe["cooking_time"]
            
            # 난이도 카운트
            difficulty_counts[recipe["difficulty"]] += 1
            
            # 영양소 누적
            nutrition = recipe["nutrition_per_100g"]
            for nutrient in nutrition_totals.keys():
                nutrition_totals[nutrient] += nutrition.get(nutrient, 0)
        
        # 결과 계산
        recipe_count = len(self.recipe_database)
        
        trends["diet_popularity"] = {diet: count/recipe_count*100 
                                   for diet, count in diet_counts.items()}
        trends["avg_cooking_time"] = total_cooking_time / recipe_count
        trends["difficulty_distribution"] = {diff: count/recipe_count*100 
                                           for diff, count in difficulty_counts.items()}
        trends["nutrition_averages"] = {nutrient: total/recipe_count 
                                      for nutrient, total in nutrition_totals.items()}
        
        return trends
    
    def display_trends(self):
        # 레시피 트렌드 정보 출력 메서드
        
        print("\n레시피 트렌드 분석 (국가표준식품성분표 기반)")
        print("=" * 60)
        
        trends = self.analyze_recipe_trends()
        
        # 식단별 인기도
        print("\n식단 유형별 분포:")
        for diet_type, percentage in trends["diet_popularity"].items():
            print(f"  • {diet_type}: {percentage:.1f}%")
        
        # 평균 조리시간
        print(f"\n평균 조리시간: {trends['avg_cooking_time']:.1f}분")
        
        # 난이도 분포
        print("\n난이도 분포:")
        for difficulty, percentage in trends["difficulty_distribution"].items():
            print(f"  • {difficulty}: {percentage:.1f}%")
        
        # 평균 영양소
        print("\n평균 영양소 (100g당):")
        nutrition_avg = trends["nutrition_averages"]
        print(f"  • 칼로리: {nutrition_avg['kcal']:.1f} kcal")
        print(f"  • 단백질: {nutrition_avg['protein_g']:.1f}g")
        print(f"  • 탄수화물: {nutrition_avg['carb_g']:.1f}g")
        print(f"  • 지방: {nutrition_avg['fat_g']:.1f}g")
        print(f"  • 당류: {nutrition_avg['당류']:.1f}g")
        print(f"  • 나트륨: {nutrition_avg['sodium_mg']:.1f}mg")


def main():
    # 메인 실행 함수
    # 프로그램의 진입점으로 추천 시스템 전체를 실행
    
    print("AI 기반 레시피 추천 시스템 시작")
    print("국가표준식품성분표 기반 정확한 영양소 분석 제공")
    print("=" * 60)
    
    # 시스템 초기화
    try:
        # 국가표준식품성분표 사용으로 변경
        recommender = RecipeRecommendationSystem('전처리_국가표준식품성분표.csv')
        
        # 메인 메뉴
        while True:
            print("\n메인 메뉴")
            print("1. 맞춤 레시피 추천받기")
            print("2. 레시피 트렌드 보기")
            print("3. 종료")
            
            choice = input("선택하세요 (1-3): ").strip()
            
            if choice == "1":
                # 추천 세션 실행
                recommender.run_recommendation_session()
                
            elif choice == "2":
                # 트렌드 분석 표시
                recommender.display_trends()
                
            elif choice == "3":
                print("\n시스템을 종료합니다. 맛있는 요리 하세요!")
                break
                
            else:
                print("\n잘못된 입력입니다. 다시 선택해주세요.")
        
    except FileNotFoundError:
        print("오류: '전처리_국가표준식품성분표.csv' 파일을 찾을 수 없습니다.")
        print("파일 경로를 확인해주세요.")
        print("국가표준식품성분표 CSV 파일이 현재 디렉토리에 있는지 확인하세요.")
        
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        print("시스템을 다시 시작해주세요.")
        print("문제가 지속되면 개발팀에 문의하세요.")


# 스크립트가 직접 실행될 때만 main() 함수 호출
# 모듈로 import될 때는 실행되지 않음
if __name__ == "__main__":
    main()