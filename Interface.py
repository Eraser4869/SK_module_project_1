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

STYLE_HTML = """
<style>
[data-testid="stDecoration"] {display: none !important;}
header[data-testid="stHeader"] {display: none !important;}
footer {display: none !important;}

.block-container {
    padding-left: 120px !important;
}
.stApp {
    background-image: url('https://hebbkx1anhila5yf.public.blob.vercel-storage.com/background1.jpg-53kiOtkFogmnT92whTKte9cZkoou0H.jpeg');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.main-box {
    background: white;
    padding: 30px;
    margin: 0 auto;
    max-width: 1280px;
    border-radius: 20px;
    box-shadow: 0 0 30px rgba(0, 0, 0, 0.1);
}
.app-header {
    background: #4a90e2;
    color: white;
    padding: 15px 20px;
    border-radius: 10px;
    font-weight: bold;
    font-size: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 30px;
}
.logo-img {
    width: 30px;
    height: 30px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 10px;
}
.ingredient-tag {
    display: inline-block;
    background: #007bff;
    color: white;
    border-radius: 10px;
    padding: 4px 10px;
    font-size: 12px;
    margin: 3px;
}
.info-tag {
    background: rgba(255,255,255,1);
    padding: 6px 12px;
    border-radius: 15px;
    font-size: 13px;
    margin: 3px;
    display: inline-block;
}
.section-title {
    font-size: 1.2em;
    font-weight: bold;
    margin: 15px 0 4px 0;
    color: black;
}
.analyzed-recipe-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #222;
}
.ingredient-item {
    display: inline-block;
    background: #f0f0f0;
    border-radius: 8px;
    padding: 5px 10px;
    margin: 5px 5px 5px 0;
    font-size: 13px;
}
.step-container {
    margin-top: 10px;
}
.step-item {
    margin-bottom: 8px;
}
div[data-testid="stColumn"] {
    padding-left: 2px !important;
    padding-right: 2px !important;
}

/* ✅ 레시피 카드 스타일 보완 */
.recipe-card {
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 16px;
    padding: 20px;
    width: 100%;
    max-width: 300px;
    min-height: 380px;
    margin: 0 auto 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    padding-top: 40px;
    position: relative;
}
.recipe-card .card-content {
    flex-grow: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    margin-bottom: 20px;
}
.recipe-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}
.recipe-image {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    object-fit: cover;
    margin-bottom: 10px;
}
.recipe-title {
    font-size: 18px;
    font-weight: 600;
    color: #333;
    margin-bottom: 8px;
    text-align: center;
}
.recipe-instructions {
    font-size: 14px;
    color: #444;
    text-align: left;
    margin-top: 10px;
    line-height: 1.6;
    white-space: pre-wrap;
}
.recipe-card .card-button {
    margin-top: auto;
    width: 100%;
    text-align: center;
}
</style>
"""
st.markdown(STYLE_HTML, unsafe_allow_html=True)



class StreamlitRecipeApp:
    """Streamlit 레시피 추천 앱"""
    
    def __init__(self):
        self.preference_options = {
            'diet': ['다이어트', '저탄고지', '저염', '채식'],
            'time': ['15분 이내', '30분 이내', '45분 이내'],
            'difficulty': ['쉬움', '보통', '어려움']
        }

        # 샘플 레시피 카드
        self.sample_recipes = [
            {
                "title": "오므라이스",
                "ingredients": "계란 1개, 밥 1공기, 양파 1/4개, 당근 1/3개",
                "instructions": [
                    "양파와 당근을 볶는다",
                    "밥을 넣고 간한다",
                    "계란을 부쳐 밥을 감싼다"
                ],
                "image_url": "https://cdn-icons-png.flaticon.com/512/1046/1046784.png",
                "key": "omurice"
            },
            {
                "title": "샐러드",
                "ingredients": "양상추, 오이, 토마토, 드레싱",
                "instructions": [
                    "채소를 씻고 썬다",
                    "그릇에 담고 드레싱을 뿌린다"
                ],
                "image_url": "https://cdn-icons-png.flaticon.com/512/135/135620.png",
                "key": "salad"
            },
            {
                "title": "토마토 파스타",
                "ingredients": "파스타면, 토마토, 양파, 마늘",
                "instructions": [
                    "파스타를 삶는다",
                    "토마토, 마늘, 양파로 소스를 만든다",
                    "면을 넣고 볶는다"
                ],
                "image_url": "https://cdn-icons-png.flaticon.com/512/857/857681.png",
                "key": "pasta"
            }
        ]

    def run(self):
        st.set_page_config(page_title="🍽️ AI 맞춤 레시피 추천", page_icon="🍽️", layout="wide")

        st.markdown("""
        <div class="app-header">
            <div style="display:flex;align-items:center;">
                AI 맞춤 레시피 추천 시스템
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1.3, 2])

        with col1:
            preferences = self._get_user_preferences()
            ingredient_data = self._get_ingredient_input()
            if st.button("맞춤 레시피 추천받기", type="primary", use_container_width=True):
                has_text = ingredient_data.get('text_ingredients')
                has_image = ingredient_data.get('image_data')
                if has_text or has_image:
                    self._save_user_input_data(preferences, ingredient_data)
                    self._generate_recommendations_from_complete_pipeline(col2)
                else:
                    st.error("❌ 텍스트 재료 또는 이미지 중 하나는 반드시 입력해주세요!")

        with col2:
            if st.session_state.get("recommendations_generated", False):
                # 실제 추천 결과 표시
                pass  # 여긴 이미 _generate_recommendations_from_complete_pipeline 안에서 처리됨
            else:
                # 추천 버튼 누르기 전 안내 메시지
                st.info("👈 왼쪽에서 설정을 완료하고 '맞춤 레시피 추천받기' 버튼을 눌러주세요!")
                self._display_sample_recipe_cards()

    def _get_user_preferences(self) -> Dict:
        """사용자 선호도 입력 받기"""
        st.subheader("🎯 선호도 설정")
        st.caption("선택하지 않으면 선호사항 없음으로 처리됩니다")

        preferences = {
            "diet": [0, 0, 0, 0],
            "time": [0, 0, 0],
            "difficulty": [0, 0, 0]
        }

        # ✅ 식단 유형 (checkbox 가로 정렬 + 간격 최소화)
        st.markdown("**🥗 식단 유형**")
        diet_options = self.preference_options["diet"]

        # 2줄로 나눔: ['다이어트', '저탄고지'], ['저염', '채식']
        rows = [diet_options[:2], diet_options[2:]]

        for row_index, row_options in enumerate(rows):
            cols = st.columns(2)
            for i, (col, label) in enumerate(zip(cols, row_options)):
                with col:
                    global_index = row_index * 2 + i  # 실제 인덱스 계산
                    preferences["diet"][global_index] = 1 if st.checkbox(label, key=f"diet_{global_index}") else 0

        st.markdown("---")

        # ✅ 조리시간 (radio + 가로 정렬)
        st.markdown("**⏰ 조리시간**")
        time_selection = st.radio(
            label="",
            options=[None] + self.preference_options["time"],
            format_func=lambda x: "선택 없음" if x is None else x,
            key="time_selection",
            horizontal=True
        )
        if time_selection:
            time_index = self.preference_options["time"].index(time_selection)
            preferences["time"][time_index] = 1

        st.markdown("---")

        # ✅ 조리 난이도 (radio + 가로 정렬)
        st.markdown("**🎯 조리 난이도**")
        difficulty_selection = st.radio(
            label="",
            options=[None] + self.preference_options["difficulty"],
            format_func=lambda x: "선택 없음" if x is None else x,
            key="difficulty_selection",
            horizontal=True
        )
        if difficulty_selection:
            diff_index = self.preference_options["difficulty"].index(difficulty_selection)
            preferences["difficulty"][diff_index] = 1

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

    def _display_sample_recipe_cards(self):
        st.markdown("### 🍲 오늘의 추천 레시피")
        cols = st.columns(3)

        for idx, recipe in enumerate(self.sample_recipes):
            title = recipe["title"]
            ingredients = recipe["ingredients"]
            instructions = recipe["instructions"]
            key = recipe["key"]

            if f"expand_{key}" not in st.session_state:
                st.session_state[f"expand_{key}"] = False

            with cols[idx]:
                # 카드 내용만 HTML로 렌더링
                html = f"""
                <div class="recipe-card">
                    <div class="card-content">
                        <div class="recipe-title">{title}</div>
                        <div style='font-size:13px; color:#555; margin-bottom:8px;'>{ingredients}</div>
                """

                # 조리법 펼치기
                if st.session_state[f"expand_{key}"]:
                    html += "<div style='text-align:left;font-size:13px;margin-top:10px;'>"
                    for i, step in enumerate(instructions, 1):
                        html += f"<p><strong>{i}.</strong> {step}</p>"
                    html += "</div>"

                html += "</div></div>"  # card-content & recipe-card 닫기
                st.markdown(html, unsafe_allow_html=True)

                # ✅ 버튼은 카드 외부에 표시
                if not st.session_state[f"expand_{key}"]:
                    if st.button(f"📖 조리법 보기", key=f"show_{key}"):
                        st.session_state[f"expand_{key}"] = True
                        st.rerun()
                else:
                    if st.button(f"🔙 조리법 접기", key=f"hide_{key}"):
                        st.session_state[f"expand_{key}"] = False
                        st.rerun()

    def _format_instructions(self, instructions):
        """조리법 리스트를 예쁘게 포맷"""
        return "\n".join([f"{i+1}. {step}" for i, step in enumerate(instructions)])
    
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
            #'timestamp': st.session_state.get('timestamp', None) or st.text_input("현재 시간", value="2024-01-01 12:00:00", disabled=True)
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
                
                #st.success(f"✅ {len(recipe_dict)}개 레시피 발견: {list(recipe_dict.keys())}")
            
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
                
                #st.success(f"✅ {len(integrated_data)}개 레시피 분석 완료")
            
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
        
        
        # 6단계: 상세 레시피 표시
        st.markdown("---")
        self._display_detailed_recipe_streamlit(best_recipe_name, best_recipe_data)
        
        # 7단계: 완료 메시지
        st.success(f"✨ '{best_recipe_name}' 레시피 추천이 완료되었습니다!")
        st.info(f"🎯 선호도 매칭률: {best_score:.0f}/100점")
        st.write("🍽️ 맛있는 요리 되세요!")
        
        st.session_state.recommendations_generated = True

    def _display_user_preferences_streamlit(self, user_preferences: Dict, recommender: RecipeRecommender):
        """사용자 선호도 정보 요약 (라운드형 태그로 출력)"""
        #st.markdown("### 👤 사용자 선호도 설정 요약:")

        # 선호 식단
        diet_prefs = user_preferences.get('diet', [0, 0, 0, 0])
        preferred_diets = [recommender.preference_mapping['diet'][i] for i, pref in enumerate(diet_prefs) if pref == 1]
        diet_display = " / ".join(preferred_diets) if preferred_diets else "없음"

        # 조리 시간
        time_prefs = user_preferences.get('time', [0, 0, 0])
        preferred_times = [recommender.preference_mapping['time'][i] for i, pref in enumerate(time_prefs) if pref == 1]
        time_display = " / ".join(preferred_times) if preferred_times else "없음"

        # 난이도
        diff_prefs = user_preferences.get('difficulty', [0, 0, 0])
        preferred_difficulties = [recommender.preference_mapping['difficulty'][i] for i, pref in enumerate(diff_prefs) if pref == 1]
        diff_display = " / ".join(preferred_difficulties) if preferred_difficulties else "없음"

        # 스타일 적용 HTML 출력
        st.markdown(f"""
        <div style="margin-bottom: 10px;">
            <span class="info-tag">🥗 {diet_display}</span>
            <span class="info-tag">⏰ {time_display}</span>
            <span class="info-tag">🎯 {diff_display}</span>
        </div>
        """, unsafe_allow_html=True)
    
    def _display_recipe_summary_streamlit(self, recipe_name: str, recipe_data: Dict, rank: int, recommender: RecipeRecommender):
        # 이 함수는 출력 생략: 카드 UI에서 제거됨
        pass

    def _display_detailed_recipe_streamlit(self, recipe_name: str, recipe_data: Dict):
        st.markdown(f"<h3>{recipe_name}</h3>", unsafe_allow_html=True)

        original_recipe = recipe_data.get('original_recipe_data', {})

        # 필요한 재료
        ingredients = original_recipe.get('재료', [])
        st.markdown('<div class="section-title">🥘 필요한 재료</div>', unsafe_allow_html=True)
        ingredients_html = "".join([
            f'<span class="ingredient-tag">{ing["item"]} {ing["amount"]}{ing["unit"]}</span>'
            for ing in ingredients
        ])
        st.markdown(f"<div>{ingredients_html}</div>", unsafe_allow_html=True)

        # 조리법
        st.markdown('<div class="section-title">👩‍🍳 조리법</div>', unsafe_allow_html=True)
        try:
            from crawling import food_info
            with st.spinner(f"'{recipe_name}' 레시피를 검색 중..."):
                detailed_recipe_web = food_info(recipe_name)
            recipe_steps = detailed_recipe_web.get('recipe', [])
            for i, step in enumerate(recipe_steps, 1):
                st.markdown(f"**{i}.** {step}")
        except Exception as e:
            st.warning("⚠️ 웹 레시피를 불러오지 못했습니다.")
            st.exception(e)


def main():
    """메인 실행 함수"""
    app = StreamlitRecipeApp()
    app.run()


if __name__ == "__main__":
    main()