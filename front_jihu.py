import streamlit as st
import base64
import json
from typing import Dict, List, Any
from PIL import Image
import io

# 기존 모듈들 import
from recipe_data_integrator import example_usage
from recipe_recommend import RecipeRecommender

# 실제 처리 모듈들 import
from GPTAPI import extract_ingredients_with_gpt
from food_ingredients_detect_module import IngredientDetector
from run_recommendation import run_recipe_recommendation

class StreamlitRecipeApp:
    """Streamlit 레시피 추천 앱"""
    
    def __init__(self):
        # 선호도 매핑
        self.preference_options = {
            'diet': ['다이어트', '저탄고지', '저염', '채식'],
            'time': ['15분 이내', '30분 이내', '45분 이내'],
            'difficulty': ['쉬움', '보통', '어려움']
        }
    
    def run(self):
        """메인 앱 실행"""
        st.set_page_config(
            page_title="🍽️ AI 맞춤 레시피 추천",
            page_icon="🍽️",
            layout="wide"
        )
        
        st.title("🍽️ AI 맞춤 레시피 추천 시스템")
        st.markdown("---")
        
        # 사이드바에 입력 폼
        with st.sidebar:
            st.header("📝 설정")
            preferences = self._get_user_preferences()
            ingredient_data = self._get_ingredient_input()
        
        # 메인 영역
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("⚙️ 입력 정보 확인")
            self._display_input_summary(preferences, ingredient_data)
            
            # 추천 버튼
            if st.button("🎯 맞춤 레시피 추천받기", type="primary", use_container_width=True):
                # 텍스트나 이미지 중 하나는 반드시 있어야 함
                has_text = ingredient_data.get('text_ingredients') and len(ingredient_data['text_ingredients']) > 0
                has_image = ingredient_data.get('image_data') is not None
                
                if has_text or has_image:
                    # 입력 데이터 저장
                    self._save_user_input_data(preferences, ingredient_data)
                    
                    # 실제 파이프라인으로 추천 생성
                    self._generate_recommendations_from_complete_pipeline(col2)
                else:
                    st.error("❌ 텍스트 재료 또는 이미지 중 하나는 반드시 입력해주세요!")
        
        with col2:
            if 'recommendations_generated' not in st.session_state:
                st.header("📋 추천 결과")
                st.info("👈 왼쪽에서 설정을 완료하고 '맞춤 레시피 추천받기' 버튼을 눌러주세요!")

    def _get_user_preferences(self) -> Dict:
        """사용자 선호도 입력 받기"""
        st.subheader("🎯 선호도 설정")
        st.caption("선택하지 않으면 선호사항 없음으로 처리됩니다")
        
        preferences = {
            "diet": [0, 0, 0, 0],
            "time": [0, 0, 0],
            "difficulty": [0, 0, 0]
        }
        
        # 식단 선호도
        st.markdown("**🥗 선호하는 식단 유형** (복수 선택 가능)")
        for i, diet_type in enumerate(self.preference_options['diet']):
            if st.checkbox(diet_type, key=f"diet_{i}"):
                preferences["diet"][i] = 1
        
        st.markdown("---")
        
        # 조리시간 선호도
        st.markdown("**⏰ 선호하는 조리시간**")
        time_selection = st.radio(
            "시간 선택",
            options=[None] + self.preference_options['time'],
            format_func=lambda x: "선택 없음" if x is None else x,
            key="time_selection"
        )
        if time_selection and time_selection != None:
            time_index = self.preference_options['time'].index(time_selection)
            preferences["time"][time_index] = 1
        
        st.markdown("---")
        
        # 난이도 선호도
        st.markdown("**🎯 선호하는 조리 난이도**")
        difficulty_selection = st.radio(
            "난이도 선택",
            options=[None] + self.preference_options['difficulty'],
            format_func=lambda x: "선택 없음" if x is None else x,
            key="difficulty_selection"
        )
        if difficulty_selection and difficulty_selection != None:
            difficulty_index = self.preference_options['difficulty'].index(difficulty_selection)
            preferences["difficulty"][difficulty_index] = 1
        
        return preferences
    
    def _get_ingredient_input(self) -> Dict:
        """재료 입력 받기 (텍스트와 이미지 모두 가능)"""
        st.subheader("🥬 보유 재료 입력")
        st.caption("텍스트 재료 또는 이미지 중 하나는 반드시 입력해주세요")
        
        ingredient_data = {}
        
        # 텍스트 재료 입력
        st.markdown("**📝 텍스트로 재료 입력**")
        
        # 세션 상태 초기화
        if 'text_ingredients' not in st.session_state:
            st.session_state.text_ingredients = [""]
        
        # 재료 입력 필드들
        text_ingredients = []
        for i, ingredient in enumerate(st.session_state.text_ingredients):
            col_input, col_remove = st.columns([4, 1])
            
            with col_input:
                new_ingredient = st.text_input(
                    f"재료 {i+1}",
                    value=ingredient,
                    key=f"text_ingredient_{i}",
                    placeholder="예: 토마토 200g"
                )
                if new_ingredient.strip():
                    text_ingredients.append(new_ingredient.strip())
            
            with col_remove:
                if len(st.session_state.text_ingredients) > 1:
                    if st.button("🗑️", key=f"remove_{i}", help="재료 삭제"):
                        st.session_state.text_ingredients.pop(i)
                        st.rerun()
        
        # 재료 추가 버튼
        if st.button("➕ 재료 추가", use_container_width=True):
            st.session_state.text_ingredients.append("")
            st.rerun()
        
        ingredient_data['text_ingredients'] = text_ingredients
        
        st.markdown("---")
        
        # 이미지 재료 입력
        st.markdown("**📷 이미지로 재료 입력**")
        
        uploaded_file = st.file_uploader(
            "재료가 포함된 이미지를 업로드해주세요",
            type=['png', 'jpg', 'jpeg'],
            key="image_upload"
        )
        
        if uploaded_file is not None:
            # 이미지 표시
            image = Image.open(uploaded_file)
            st.image(image, caption="업로드된 이미지", use_container_width=True)
            
            # base64로 인코딩
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            ingredient_data['image_data'] = img_base64
            ingredient_data['image_filename'] = uploaded_file.name
        
        return ingredient_data

    def _display_input_summary(self, preferences: Dict, ingredient_data: Dict):
        """입력 정보 요약 표시"""
        st.subheader("📊 설정 요약")
        
        # 선호도 요약
        st.markdown("**🎯 선호도 설정:**")
        
        # 식단
        selected_diets = [self.preference_options['diet'][i] for i, pref in enumerate(preferences['diet']) if pref == 1]
        if selected_diets:
            st.write(f"🥗 식단: {', '.join(selected_diets)}")
        else:
            st.write("🥗 식단: 선택 없음")
        
        # 시간
        selected_times = [self.preference_options['time'][i] for i, pref in enumerate(preferences['time']) if pref == 1]
        if selected_times:
            st.write(f"⏰ 시간: {', '.join(selected_times)}")
        else:
            st.write("⏰ 시간: 선택 없음")
        
        # 난이도
        selected_difficulties = [self.preference_options['difficulty'][i] for i, pref in enumerate(preferences['difficulty']) if pref == 1]
        if selected_difficulties:
            st.write(f"🎯 난이도: {', '.join(selected_difficulties)}")
        else:
            st.write("🎯 난이도: 선택 없음")
        
        st.markdown("---")
        
        # 재료 요약
        st.markdown("**🥬 재료 정보:**")
        
        # 텍스트 재료
        text_ingredients = ingredient_data.get('text_ingredients', [])
        if text_ingredients:
            st.write(f"📝 텍스트 재료: {len(text_ingredients)}개")
            for i, ingredient in enumerate(text_ingredients, 1):
                st.write(f"  {i}. {ingredient}")
        else:
            st.write("📝 텍스트 재료: 없음")
        
        # 이미지
        if ingredient_data.get('image_data'):
            st.write(f"📷 이미지: {ingredient_data.get('image_filename', '업로드됨')}")
        else:
            st.write("📷 이미지: 없음")
    
    def _save_user_input_data(self, preferences: Dict, ingredient_data: Dict):
        """사용자 입력 데이터를 저장"""
        st.session_state.user_input_data = {
            'preferences': preferences,
            'ingredient_data': ingredient_data,
            'timestamp': st.session_state.get('timestamp', None) or st.text_input("현재 시간", value="2024-01-01 12:00:00", disabled=True)
        }

    def _generate_recommendations_from_complete_pipeline(self, output_column):
        """완전한 파이프라인으로 추천 생성 및 표시"""
        try:
            # 사용자 입력 데이터 가져오기
            user_data = st.session_state.user_input_data
            
            with st.spinner("🔄 입력 데이터 처리 중..."):
                # Step 1: 텍스트 재료 처리 (GPTAPI.py)
                text_ingredients = []
                if user_data['ingredient_data'].get('text_ingredients'):
                    text_input = " ".join(user_data['ingredient_data']['text_ingredients'])
                    if text_input.strip():
                        st.info(f"📝 텍스트 입력: {text_input}")
                        text_ingredients = extract_ingredients_with_gpt(text_input)
                        st.success(f"✅ 텍스트에서 {len(text_ingredients)}개 재료 추출: {text_ingredients}")
                
                # Step 2: 이미지 재료 처리 (food_ingredients_detect_module.py)
                image_ingredients = []
                if user_data['ingredient_data'].get('image_data'):
                    detector = IngredientDetector()
                    image_dict = {
                        "mode": "RGB", 
                        "size": [400, 400],
                        "data": user_data['ingredient_data']['image_data']
                    }
                    image_ingredients = detector.return_ingredient([image_dict])
                    st.success(f"✅ 이미지에서 {len(image_ingredients)}개 재료 추출: {image_ingredients}")
                
                # Step 3: 재료 통합
                all_ingredients = list(set(text_ingredients + image_ingredients))
                if not all_ingredients:
                    st.error("❌ 추출된 재료가 없습니다.")
                    return
                
                st.info(f"🥬 최종 통합 재료 {len(all_ingredients)}개: {', '.join(all_ingredients)}")
            
            with st.spinner("🍽️ 맞춤 레시피 검색 중..."):
                # Step 4: 레시피 추천 (run_recommendation.py)
                recipe_dict = run_recipe_recommendation(all_ingredients)
                if not recipe_dict:
                    st.error("❌ 해당 재료로 만들 수 있는 레시피를 찾지 못했습니다.")
                    return
                
                st.success(f"✅ {len(recipe_dict)}개 레시피 발견: {list(recipe_dict.keys())}")
            
            with st.spinner("🧠 AI 분석 및 통합 중..."):
                # Step 5: 데이터 통합 및 분석 (recipe_data_integrator.py)
                # 전역 변수에 실제 레시피 데이터 설정 후 example_usage 호출
                from recipe_data_integrator import set_recipe_data, example_usage
                
                # 실제 레시피 데이터를 전역 변수에 설정
                set_recipe_data(recipe_dict)
                
                # 기존 example_usage 그대로 호출 (내부에서 전역 변수 사용)
                integrated_data, processed_preferences = example_usage(user_data['preferences'])
                
                if not integrated_data:
                    st.error("❌ 레시피 분석에 실패했습니다.")
                    return
                
                st.success(f"✅ {len(integrated_data)}개 레시피 분석 완료")
            
            # Step 6: 결과 표시
            with output_column:
                self._display_complete_recommendations(integrated_data, processed_preferences)
                
        except Exception as e:
            st.error(f"❌ 파이프라인 처리 중 오류: {e}")
            with st.expander("상세 오류 정보"):
                st.exception(e)



    def _display_complete_recommendations(self, integrated_data: Dict, user_preferences: Dict):
        """완전한 추천 결과 표시"""
        st.header("🎉 AI 맞춤 레시피 추천 결과!")
        st.write("🎯 사용자 선호도를 기반으로 최적의 레시피를 추천해드립니다.")
        
        # 추천기 초기화
        recommender = RecipeRecommender()
        
        # 1단계: 사용자 선호도 표시
        self._display_user_preferences_streamlit(user_preferences, recommender)
        
        # 2단계: 추천 생성 및 분석
        recommendations = recommender.get_recommendations_by_preference(
            integrated_data, user_preferences, len(integrated_data)
        )
        
        if not recommendations:
            st.error("❌ 죄송합니다. 조건에 맞는 레시피를 찾을 수 없습니다.")
            return
        
        st.success(f"🔍 총 {len(recommendations)}개의 레시피를 분석했습니다!")
        
        # 3단계: 가장 높은 점수의 레시피 선택
        best_recipe_name, best_recipe_data = recommendations[0]
        best_score = best_recipe_data.get('preference_score', 0)
        
        st.info("🏆 선호도 점수가 가장 높은 레시피를 추천드립니다!")
        
        # 4단계: 레시피 요약 정보 표시
        self._display_recipe_summary_streamlit(best_recipe_name, best_recipe_data, 1, recommender)
        
        # 5단계: 다른 추천 레시피들도 간단히 표시
        if len(recommendations) > 1:
            st.markdown("### 📋 다른 추천 레시피들:")
            for i, (recipe_name, recipe_data) in enumerate(recommendations[1:], 2):
                score = recipe_data.get('preference_score', 0)
                st.write(f"  **#{i}. {recipe_name}** (점수: {score:.0f}/100)")
        
        # 6단계: 상세 레시피 표시
        st.markdown("---")
        self._display_detailed_recipe_streamlit(best_recipe_name, best_recipe_data)
        
        # 7단계: 완료 메시지
        st.success(f"✨ '{best_recipe_name}' 레시피 추천이 완료되었습니다!")
        st.info(f"🎯 선호도 매칭률: {best_score:.0f}/100점")
        st.write("🍽️ 맛있는 요리 되세요!")
        
        st.session_state.recommendations_generated = True

    def _display_user_preferences_streamlit(self, user_preferences: Dict, recommender: RecipeRecommender):
        """사용자 선호도 정보 출력"""
        st.markdown("### 👤 사용자 선호도 설정:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # 식단 선호도
            diet_prefs = user_preferences.get('diet', [0, 0, 0, 0])
            preferred_diets = [recommender.preference_mapping['diet'][i] for i, pref in enumerate(diet_prefs) if pref == 1]
            if preferred_diets:
                st.write(f"🥗 **선호 식단:** {', '.join(preferred_diets)}")
            else:
                st.write(f"🥗 **선호 식단:** 없음")
        
        with col2:
            # 시간 선호도
            time_prefs = user_preferences.get('time', [0, 0, 0])
            preferred_times = [recommender.preference_mapping['time'][i] for i, pref in enumerate(time_prefs) if pref == 1]
            if preferred_times:
                st.write(f"⏰ **선호 조리시간:** {', '.join(preferred_times)}")
            else:
                st.write(f"⏰ **선호 조리시간:** 없음")
        
        with col3:
            # 난이도 선호도
            difficulty_prefs = user_preferences.get('difficulty', [0, 0, 0])
            preferred_difficulties = [recommender.preference_mapping['difficulty'][i] for i, pref in enumerate(difficulty_prefs) if pref == 1]
            if preferred_difficulties:
                st.write(f"🎯 **선호 난이도:** {', '.join(preferred_difficulties)}")
            else:
                st.write(f"🎯 **선호 난이도:** 없음")
    
    def _display_recipe_summary_streamlit(self, recipe_name: str, recipe_data: Dict, rank: int, recommender: RecipeRecommender):
        """레시피 요약 정보 출력"""
        preference_score = recipe_data.get('preference_score', 0)
        
        st.markdown(f"### 🏆 추천 레시피 #{rank}: {recipe_name} (선호도 점수: {preference_score:.0f}/100)")
        
        # 선호도 매칭 정보
        preference_matches = recipe_data.get('preference_matches', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 식단 매칭
            diet_matches = preference_matches.get('diet_matches', [])
            if diet_matches:
                st.write(f"✅ **식단 매칭:** {', '.join(diet_matches)}")
            else:
                st.write(f"❌ **식단 매칭:** 없음")
            
            # 시간 매칭
            time_match = preference_matches.get('time_match', False)
            cooking_info = recipe_data.get('cooking_info', {})
            predicted_time = cooking_info.get('predicted_time_minutes', 30)
            if time_match:
                st.write(f"✅ **시간 매칭:** {predicted_time:.0f}분 (선호 시간대와 일치)")
            else:
                st.write(f"❌ **시간 매칭:** {predicted_time:.0f}분 (선호 시간대와 불일치)")
        
        with col2:
            # 난이도 매칭
            difficulty_match = preference_matches.get('difficulty_match', False)
            difficulty_level = cooking_info.get('difficulty_level', '보통')
            if difficulty_match:
                st.write(f"✅ **난이도 매칭:** {difficulty_level} (선호 난이도와 일치)")
            else:
                st.write(f"❌ **난이도 매칭:** {difficulty_level} (선호 난이도와 불일치)")
            
            # 식단 분류 정보
            diet_classifications = recipe_data.get('diet_classifications', {})
            applicable_diets = [diet for diet, applicable in diet_classifications.items() if applicable]
            if applicable_diets:
                st.write(f"📋 **식단 유형:** {', '.join(applicable_diets)}")
            else:
                st.write("📋 **식단 유형:** 일반식")
        
        # 영양 정보
        nutrition_info = recipe_data.get('nutrition_info', {})
        if nutrition_info:
            nutrition_per_100g = nutrition_info.get('nutrition_per_100g', {})
            if nutrition_per_100g:
                st.write(f"💪 **영양정보(100g당):** {nutrition_per_100g.get('kcal', 0):.0f}kcal, "
                        f"단백질 {nutrition_per_100g.get('protein_g', 0):.1f}g")
    
    def _display_detailed_recipe_streamlit(self, recipe_name: str, recipe_data: Dict):
        """상세 레시피 정보 출력"""
        st.markdown(f"### 📖 상세 정보: {recipe_name}")
        
        # 원본 레시피 정보
        original_recipe = recipe_data.get('original_recipe_data', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 재료 정보
            ingredients = original_recipe.get('재료', [])
            if ingredients:
                st.markdown("**🥬 재료:**")
                for ingredient in ingredients:
                    st.write(f"• {ingredient['item']}: {ingredient['amount']}{ingredient['unit']}")
            else:
                st.write("**🥬 재료:** 정보 없음")
        
        with col2:
            # 조미료 정보
            seasonings = original_recipe.get('조미료', [])
            if seasonings:
                st.markdown("**🧂 조미료:**")
                for seasoning in seasonings:
                    st.write(f"• {seasoning['item']}: {seasoning['amount']}{seasoning['unit']}")
            else:
                st.write("**🧂 조미료:** 정보 없음")
        
        # 식단 분류 상세 정보
        diet_classifications = recipe_data.get('diet_classifications', {})
        diet_reasons = recipe_data.get('diet_reasons', {})
        
        st.markdown("**📊 식단 분류 상세:**")
        for diet_type, applicable in diet_classifications.items():
            status = "✅ 적합" if applicable else "❌ 부적합"
            reason = diet_reasons.get(diet_type, "")
            st.write(f"• **{diet_type}:** {status}")
            if reason:
                st.caption(f"  💬 이유: {reason}")
        
        # 웹에서 상세 레시피 가져오기
        st.markdown("---")
        st.markdown("### 🌐 웹에서 상세 레시피 가져오는 중...")
        
        try:
            from crawling import food_info
            
            with st.spinner(f"'{recipe_name}' 레시피를 검색 중..."):
                detailed_recipe_web = food_info(recipe_name)
            
            if detailed_recipe_web:
                st.success("✅ 상세 레시피를 성공적으로 가져왔습니다!")
                
                # 웹에서 가져온 재료 정보
                st.markdown("### 🍳 상세 레시피:")
                st.markdown("**📝 재료 목록:**")
                ingredients_text = detailed_recipe_web.get('ingredients', '정보 없음')
                st.text_area("재료", ingredients_text, height=100, disabled=True)
                
                # 웹에서 가져온 조리 방법
                st.markdown("**👨‍🍳 조리 방법:**")
                recipe_steps = detailed_recipe_web.get('recipe', [])
                if recipe_steps:
                    for i, step in enumerate(recipe_steps, 1):
                        st.write(f"**{i}.** {step}")
                else:
                    st.write("조리 방법 정보가 없습니다.")
            else:
                st.warning(f"⚠️ 웹에서 '{recipe_name}' 레시피를 찾을 수 없습니다.")
                st.info("💡 레시피명을 조금 더 구체적으로 입력하시거나, 다른 레시피를 시도해보세요.")
                
        except ImportError:
            st.error("❌ 크롤링 모듈을 찾을 수 없습니다. crawling.py 파일이 있는지 확인해주세요.")
        except Exception as e:
            st.error(f"❌ 웹 레시피 가져오기 실패: {e}")
            with st.expander("오류 상세 정보"):
                st.exception(e)


def main():
    """메인 실행 함수"""
    app = StreamlitRecipeApp()
    app.run()


if __name__ == "__main__":
    main()