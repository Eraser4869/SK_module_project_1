# 간소화된 레시피 분류기 - 즉시 사용 가능한 버전
# 머신러닝 없이 규칙 기반으로만 구현

import pandas as pd
import json
from typing import Dict, List, Tuple, Optional

class SimpleRecipeClassifier:
    """규칙 기반 간단한 레시피 분류기"""
    
    def __init__(self, nutrition_csv_path: str):
        self.nutrition_db = self._load_nutrition_data(nutrition_csv_path)
        self.classification_rules = self._define_classification_rules()
        
    def _load_nutrition_data(self, csv_path: str) -> pd.DataFrame:
        """영양소 데이터베이스 로드"""
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df = df.dropna(subset=['식품명'])
        df.set_index('식품명', inplace=True)
        
        # 기본 재료 추가
        basic_ingredients = {
            '물': {'kcal': 0, 'carb_g': 0, 'fat_g': 0, 'protein_g': 0, 'sodium_mg': 0, 'sugar_g': 0},
            '소금': {'kcal': 0, 'carb_g': 0, 'fat_g': 0, 'protein_g': 0, 'sodium_mg': 40000, 'sugar_g': 0},
            '설탕': {'kcal': 387, 'carb_g': 99.8, 'fat_g': 0, 'protein_g': 0, 'sodium_mg': 1, 'sugar_g': 99.8},
            '올리브오일': {'kcal': 884, 'carb_g': 0, 'fat_g': 100, 'protein_g': 0, 'sodium_mg': 2, 'sugar_g': 0},
            '참기름': {'kcal': 884, 'carb_g': 0, 'fat_g': 100, 'protein_g': 0, 'sodium_mg': 0, 'sugar_g': 0},
            '간장': {'kcal': 50, 'carb_g': 5, 'fat_g': 0, 'protein_g': 8, 'sodium_mg': 5000, 'sugar_g': 5},
            '고춧가루': {'kcal': 282, 'carb_g': 56, 'fat_g': 12, 'protein_g': 12, 'sodium_mg': 35, 'sugar_g': 28}
        }
        
        for ingredient, nutrition in basic_ingredients.items():
            if ingredient not in df.index:
                df.loc[ingredient] = {col: nutrition.get(col.split('_')[0], 0) for col in df.columns}
        
        return df
    
    def _define_classification_rules(self) -> Dict:
        """분류 규칙 정의 (100g 기준)"""
        return {
            '다이어트': {
                'kcal_max': 250,
                'fat_max': 10,
                'protein_min': 8,
                'sugar_max': 15
            },
            '저탄고지': {
                'carb_max': 15,
                'fat_min': 10,
                'protein_min': 10,
                'sugar_max': 8
            },
            '저염': {
                'sodium_max': 400
            },
            '채식': {
                'exclude_keywords': [
                    '돼지', '소', '닭', '오리', '양', '염소', '생선', '연어', '참치', 
                    '고등어', '새우', '게', '조개', '굴', '문어', '오징어', '멸치',
                    '삼겹살', '갈비', '등심', '치킨', '햄', '소시지', '베이컨'
                ]
            }
        }
    
    def find_ingredient(self, ingredient_name: str) -> Optional[str]:
        """재료명 매칭 (간단한 문자열 매칭)"""
        ingredient_name = ingredient_name.strip()
        
        # 정확히 일치
        if ingredient_name in self.nutrition_db.index:
            return ingredient_name
        
        # 부분 문자열 매칭
        for db_ingredient in self.nutrition_db.index:
            if ingredient_name in db_ingredient or db_ingredient in ingredient_name:
                return db_ingredient
        
        # 일반적인 동의어 매칭
        synonyms = {
            '닭가슴살': '닭고기',
            '삼겹살': '돼지고기',
            '소고기': '쇠고기',
            '양파': '양파_생것',
            '마늘': '마늘_생것',
            '당근': '당근_생것',
            '브로콜리': '브로콜리_생것'
        }
        
        if ingredient_name in synonyms:
            synonym = synonyms[ingredient_name]
            if synonym in self.nutrition_db.index:
                return synonym
        
        return None
    
    def convert_to_grams(self, amount: float, unit: str) -> float:
        """단위를 그램으로 변환"""
        unit = unit.lower().strip()
        
        # 부피 → 무게 변환 (근사치)
        volume_to_weight = {
            'ml': 1, 'cc': 1,
            'l': 1000,
            '큰술': 15, '작은술': 5,
            '컵': 200, '국자': 50
        }
        
        # 무게 단위
        weight_units = {
            'g': 1, 'kg': 1000, 'mg': 0.001
        }
        
        if unit in volume_to_weight:
            return amount * volume_to_weight[unit]
        elif unit in weight_units:
            return amount * weight_units[unit]
        else:
            return amount  # 기본적으로 g으로 가정
    
    def calculate_nutrition(self, recipe_data: Dict) -> Dict[str, float]:
        """레시피의 총 영양소 계산"""
        total_nutrition = {
            'kcal': 0, 'carb_g': 0, 'fat_g': 0, 
            'protein_g': 0, 'sodium_mg': 0, 'sugar_g': 0
        }
        
        matched_count = 0
        total_ingredients = 0
        
        # 재료 + 조미료 처리
        all_ingredients = recipe_data.get('재료', []) + recipe_data.get('조미료', [])
        
        for ingredient in all_ingredients:
            total_ingredients += 1
            
            item_name = ingredient.get('item', '')
            amount = ingredient.get('amount', 0)
            unit = ingredient.get('unit', 'g')
            
            if not item_name or amount <= 0:
                continue
            
            # 재료 매칭
            matched_ingredient = self.find_ingredient(item_name)
            if not matched_ingredient:
                print(f"매칭되지 않은 재료: {item_name}")
                continue
            
            matched_count += 1
            
            # 단위 변환
            amount_g = self.convert_to_grams(amount, unit)
            
            # 영양소 계산 (100g 기준 → 실제 사용량)
            ingredient_nutrition = self.nutrition_db.loc[matched_ingredient]
            multiplier = amount_g / 100
            
            for nutrient in total_nutrition.keys():
                if nutrient in ingredient_nutrition.index:
                    value = ingredient_nutrition[nutrient]
                    if pd.notna(value):
                        total_nutrition[nutrient] += value * multiplier
        
        # 매칭률 출력
        if total_ingredients > 0:
            match_rate = matched_count / total_ingredients * 100
            print(f"재료 매칭률: {match_rate:.1f}% ({matched_count}/{total_ingredients})")
        
        return total_nutrition
    
    def classify_recipe(self, recipe_name: str, recipe_data: Dict) -> Dict:
        """레시피 분류"""
        print(f"\n '{recipe_name}' 분석 중...")
        
        # 영양소 계산
        nutrition = self.calculate_nutrition(recipe_data)
        
        # 분류 수행
        classifications = {}
        
        # 다이어트 분류
        diet_rules = self.classification_rules['다이어트']
        is_diet = (
            nutrition['kcal'] <= diet_rules['kcal_max'] and
            nutrition['fat_g'] <= diet_rules['fat_max'] and
            nutrition['protein_g'] >= diet_rules['protein_min'] and
            nutrition['sugar_g'] <= diet_rules['sugar_max']
        )
        classifications['다이어트'] = is_diet
        
        # 저탄고지 분류
        keto_rules = self.classification_rules['저탄고지']
        is_keto = (
            nutrition['carb_g'] <= keto_rules['carb_max'] and
            nutrition['fat_g'] >= keto_rules['fat_min'] and
            nutrition['protein_g'] >= keto_rules['protein_min'] and
            nutrition['sugar_g'] <= keto_rules['sugar_max']
        )
        classifications['저탄고지'] = is_keto
        
        # 저염 분류
        low_sodium_rules = self.classification_rules['저염']
        is_low_sodium = nutrition['sodium_mg'] <= low_sodium_rules['sodium_max']
        classifications['저염'] = is_low_sodium
        
        # 채식 분류
        veg_rules = self.classification_rules['채식']
        recipe_text = recipe_name + ' ' + ' '.join([
            ing.get('item', '') for ing in recipe_data.get('재료', []) + recipe_data.get('조미료', [])
        ])
        is_vegetarian = not any(keyword in recipe_text for keyword in veg_rules['exclude_keywords'])
        classifications['채식'] = is_vegetarian
        
        return {
            'recipe_name': recipe_name,
            'nutrition': nutrition,
            'classifications': classifications
        }
    
    def print_result(self, result: Dict):
        """결과를 보기 좋게 출력"""
        print(f"\n {result['recipe_name']}")
        print("=" * 50)
        
        # 영양소 정보
        nutrition = result['nutrition']
        print("영양소 정보:")
        print(f"   칼로리: {nutrition['kcal']:.1f} kcal")
        print(f"   탄수화물: {nutrition['carb_g']:.1f}g")
        print(f"   단백질: {nutrition['protein_g']:.1f}g")
        print(f"   지방: {nutrition['fat_g']:.1f}g")
        print(f"   나트륨: {nutrition['sodium_mg']:.1f}mg")
        print(f"   당류: {nutrition['sugar_g']:.1f}g")
        
        # 분류 결과
        print("\n 분류 결과:")
        classifications = result['classifications']
        for label, is_classified in classifications.items():
            status = "yes" if is_classified else "no"
            print(f"   {status} {label}")
        
        # 해당하는 분류만 요약
        applicable_labels = [label for label, classified in classifications.items() if classified]
        if applicable_labels:
            print(f"\n적합한 식단: {', '.join(applicable_labels)}")
        else:
            print(f"\n적합한 식단: 일반식")


# ===== 사용 예시 =====

def main():
    """메인 실행 함수"""
    print("간소화된 레시피 분류기")
    print("=" * 50)
    
    # 분류기 초기화
    classifier = SimpleRecipeClassifier('전처리_음식DB.csv')
    
    # 테스트 레시피
    test_recipes = {
        "닭가슴살 샐러드": {
            "재료": [
                {"item": "닭가슴살", "amount": 120, "unit": "g"},
                {"item": "상추", "amount": 80, "unit": "g"},
                {"item": "토마토", "amount": 50, "unit": "g"}
            ],
            "조미료": [
                {"item": "올리브오일", "amount": 5, "unit": "ml"},
                {"item": "소금", "amount": 0.5, "unit": "g"}
            ]
        },
        "삼겹살 김치찌개": {
            "재료": [
                {"item": "삼겹살", "amount": 200, "unit": "g"},
                {"item": "김치", "amount": 150, "unit": "g"},
                {"item": "두부", "amount": 100, "unit": "g"}
            ],
            "조미료": [
                {"item": "간장", "amount": 20, "unit": "ml"},
                {"item": "고춧가루", "amount": 5, "unit": "g"},
                {"item": "물", "amount": 500, "unit": "ml"}
            ]
        },
        "현미밥": {
            "재료": [
                {"item": "현미", "amount": 100, "unit": "g"},
                {"item": "물", "amount": 150, "unit": "ml"}
            ],
            "조미료": []
        }
    }
    
    # 각 레시피 분류
    results = []
    for recipe_name, recipe_data in test_recipes.items():
        result = classifier.classify_recipe(recipe_name, recipe_data)
        classifier.print_result(result)
        results.append(result)
    
    # 전체 요약
    print("\n" + "=" * 50)
    print("분류 결과 요약")
    print("=" * 50)
    
    for result in results:
        name = result['recipe_name']
        labels = [label for label, classified in result['classifications'].items() if classified]
        labels_str = ", ".join(labels) if labels else "일반식"
        kcal = result['nutrition']['kcal']
        print(f"{name:15s} → {labels_str:15s} ({kcal:.0f} kcal)")



main()