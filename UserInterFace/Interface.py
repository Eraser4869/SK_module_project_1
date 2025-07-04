import streamlit as st
from PIL import Image
from io import BytesIO
import base64
from typing import Dict, List, Union

from input_module import get_ingredients_input, get_preferences_input


# ======================================================================
# 0. 입력 수집 모듈
# ======================================================================
def get_user_input() -> Dict[str, Dict[str, Union[List[str], List[Image.Image]]]]:
    """재료·선호 입력을 dict로 반환"""
    text_ingredients = st.session_state.get("ingredients_list", [])
    image_ingredients = []
    if st.session_state.get("uploaded_image"):
        image_ingredients.append(st.session_state["uploaded_image"])

    diet_opts = ["다이어트", "채식", "저염", "저탄고지"]
    time_opts = ["간단", "보통", "정성", "상관없음"]
    diff_opts = ["쉬움", "보통", "어려움", "상관없음"]

    diet_bools = [st.session_state.get(f"diet_{i}", False) for i in range(len(diet_opts))]
    time_bools = [st.session_state.get(f"time_{i}", False) for i in range(len(time_opts))]
    diff_bools = [st.session_state.get(f"diff_{i}", False) for i in range(len(diff_opts))]

    if not text_ingredients and not image_ingredients:
        return None

    return {
        "ingredients": {
            "text": text_ingredients,
            "image": image_ingredients,
        },
        "preferences": {
            "diet": diet_bools,
            "time": time_bools,
            "difficulty": diff_bools,
        },
    }

def add_ingredients_without_duplicates(new_ingredients_str: str):
    if not new_ingredients_str.strip():
        st.session_state.last_message = "추가할 재료를 입력해주세요."
        st.session_state.message_type = "info"
        return
    new_ingredients = [i.strip() for i in new_ingredients_str.split(',') if i.strip()]
    existing_ingredients_lower = [ingredient.lower() for ingredient in st.session_state.ingredients_list]
    unique_ingredients = []
    duplicate_ingredients = []
    for ingredient in new_ingredients:
        if ingredient.lower() not in existing_ingredients_lower:
            unique_ingredients.append(ingredient)
            existing_ingredients_lower.append(ingredient.lower())
        else:
            duplicate_ingredients.append(ingredient)
    if unique_ingredients and duplicate_ingredients:
        st.session_state.last_message = f"✅ 추가된 재료: {', '.join(unique_ingredients)}\n⚠️ 이미 추가된 재료: {', '.join(duplicate_ingredients)}"
        st.session_state.message_type = "mixed"
        st.session_state.ingredients_list.extend(unique_ingredients)
    elif unique_ingredients:
        st.session_state.last_message = f"✅ 추가된 재료: {', '.join(unique_ingredients)}"
        st.session_state.message_type = "success"
        st.session_state.ingredients_list.extend(unique_ingredients)
    elif duplicate_ingredients:
        st.session_state.last_message = f"⚠️ 이미 추가된 재료: {', '.join(duplicate_ingredients)}"
        st.session_state.message_type = "warning"
    else:
        st.session_state.last_message = "추가할 재료를 입력해주세요."
        st.session_state.message_type = "info"

st.set_page_config(page_title="AI 냉장고 요리사", page_icon="👩‍🍳", layout="wide")

st.markdown("""
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
footer, .stDeployButton, div[data-testid="stToolbar"] {
    display: none !important;
}
html, body, .stApp {
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden !important;
    width: 100% !important;
}
.main-box {
    background: white;
    padding: 30px;
    margin: 0 auto;
    margin-left: 100px;
    transform: translateX(40px);
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
.recipe-card {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 12px;
    text-align: center;
    padding: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 10px;
    width: 280px;
    height: 420px;
    display: flex;
    flex-direction: column;
    align-items: center;
}
.recipe-title {
    font-weight: bold;
    font-size: 18px;
    color: #333;
    margin-top: 10px;
    margin-bottom: 6px;
    text-align: center;
}
.recipe-image {
    width: 120px;
    height: 120px;
    object-fit: cover;
    border-radius: 50%;
    margin: 20px auto 10px;
}
.recipe-row > div {
    padding-left: 3px !important;
    padding-right: 3px !important;
}
.ingredient-circle-with-image,
.ingredient-circle-empty {
    width: 200px;
    height: 200px;
    border: 3px dashed #ccc;
    border-radius: 50%;
    margin: 0 auto 20px;
    overflow: hidden;
    position: relative;
    background: white;
    display: flex;
    justify-content: center;
    align-items: center;
    text-align: center;
    color: #666;
    font-size: 14px;
    line-height: 1.4;
}
.ingredient-image {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
}
.ingredient-box {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.preference-section {
    background: #f8f9fa;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 15px;
    border: 1px solid #e9ecef;
}
.preference-title {
    font-size: 13px;
    font-weight: bold;
    color: #495057;
    margin-bottom: 8px;
    text-align: left;
    margin-left: 20px;
}
.recipe-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 30px;
    color: white;
    margin-bottom: 20px;
    max-width: 1100px;
}
.recipe-content {
    padding: 30px;
}
.analyzed-recipe-title {
    font-size: 2.2em;
    font-weight: bold;
    margin-bottom: 15px;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
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
.ingredient-item {
    background: rgba(255,255,255,1);
    padding: 6px 12px;
    border-radius: 15px;
    font-size: 13px;
    margin: 3px;
    display: inline-block;
}
.step-container {
    background: rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 4px;
    margin: 6px 0;
}
.decision-buttons {
    text-align: center;
    margin-top: 15px;
    padding-top: 10px;
    border-top: 1px solid rgba(255,255,255,0.2);
    font-size: 14px;
}
button[kind="secondary"] {
    font-size: 15px !important;
    padding: 8px 20px !important;
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

analyzed_recipes = [
    {
        "title": "계란볶음밥",
        "image": "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dog.jpg-FYr8K5lDgiYhZVWUA1kfs3vf5fSfO3.jpeg",
        "cook_time": "15분",
        "difficulty": "쉬움",
        "calories": "420kcal",
        "ingredients": ["계란 2개", "밥 1공기", "양파 1/4개", "당근 1/3개", "간장 1큰술", "참기름 1작은술"],
        "instructions": [
            "양파와 당근을 잘게 썰어주세요.",
            "팬에 기름을 두르고 양파와 당근을 볶아주세요.",
            "계란을 풀어서 스크램블을 만들어주세요.",
            "밥을 넣고 함께 볶아주세요.",
            "간장과 참기름으로 간을 맞춰주세요."
        ]
    },
    {
        "title": "야채 오믈렛",
        "image": "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dog.jpg-FYr8K5lDgiYhZVWUA1kfs3vf5fSfO3.jpeg",
        "cook_time": "20분",
        "difficulty": "보통",
        "calories": "280kcal",
        "ingredients": ["계란 3개", "양파 1/4개", "당근 1/4개", "우유 2큰술", "소금", "후추"],
        "instructions": [
            "야채를 잘게 썰어 볶아주세요.",
            "계란에 우유를 넣고 잘 풀어주세요.",
            "팬에 계란물을 부어주세요.",
            "볶은 야채를 올리고 반으로 접어주세요.",
            "접시에 예쁘게 담아주세요."
        ]
    },
    {
        "title": "야채 스프",
        "image": "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dog.jpg-FYr8K5lDgiYhZVWUA1kfs3vf5fSfO3.jpeg",
        "cook_time": "30분",
        "difficulty": "쉬움",
        "calories": "150kcal",
        "ingredients": ["양파 1개", "당근 1개", "감자 1개", "물 500ml", "소금", "후추"],
        "instructions": [
            "모든 야채를 깍둑썰기 해주세요.",
            "팬에 야채를 볶아주세요.",
            "물을 넣고 끓여주세요.",
            "야채가 부드러워질 때까지 끓여주세요.",
            "소금과 후추로 간을 맞춰주세요."
        ]
    }
]

if 'ingredients_list' not in st.session_state:
    st.session_state.ingredients_list = []
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'show_omurice_recipe' not in st.session_state:
    st.session_state.show_omurice_recipe = False
if 'show_salad_recipe' not in st.session_state:
    st.session_state.show_salad_recipe = False
if 'show_pasta_recipe' not in st.session_state:
    st.session_state.show_pasta_recipe = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'analyzing_mode' not in st.session_state:
    st.session_state.analyzing_mode = False
if 'current_recipe_index' not in st.session_state:
    st.session_state.current_recipe_index = 0
if 'selected_recipe_mode' not in st.session_state:
    st.session_state.selected_recipe_mode = False
if 'user_input' not in st.session_state:
    st.session_state.user_input = None
if 'ingredient_input_value' not in st.session_state:
    st.session_state.ingredient_input_value = ""
if 'last_message' not in st.session_state:
    st.session_state.last_message = None
if 'message_type' not in st.session_state:
    st.session_state.message_type = None

def reset_all_states():
    st.session_state.ingredients_list = []
    st.session_state.selected_category = None
    st.session_state.show_omurice_recipe = False
    st.session_state.show_salad_recipe = False
    st.session_state.show_pasta_recipe = False
    st.session_state.uploaded_image = None
    st.session_state.image_uploaded = False
    st.session_state.analyzing_mode = False
    st.session_state.current_recipe_index = 0
    st.session_state.selected_recipe_mode = False
    st.session_state.user_input = None
    st.session_state.ingredient_input_value = ""
    st.session_state.last_message = None
    st.session_state.message_type = None

    for i in range(4):
        st.session_state[f"diet_{i}"] = False
        st.session_state[f"time_{i}"] = False
        st.session_state[f"diff_{i}"] = False

def start_analyzing():
    st.session_state.analyzing_mode = True
    st.session_state.selected_recipe_mode = False
    st.session_state.current_recipe_index = 0

def handle_yes():
    st.session_state.analyzing_mode = False
    st.session_state.selected_recipe_mode = True
    st.success("🎉 레시피를 선택하셨습니다! 요리를 시작해보세요.")
    st.balloons()

def handle_no():
    if st.session_state.current_recipe_index < len(analyzed_recipes) - 1:
        st.session_state.current_recipe_index += 1
    else:
        st.warning("더 이상 추천할 레시피가 없습니다.")
        st.session_state.analyzing_mode = False

col_header1, col_header2 = st.columns([6, 1])
with col_header1:
    st.markdown("""
    <div class="app-header">
        <div style="display:flex;align-items:center;">
            <img src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dog.jpg-FYr8K5lDgiYhZVWUA1kfs3vf5fSfO3.jpeg" class="logo-img">
            AI 냉장고 요리사
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_header2:
    if st.button("🏠 홈"):
        reset_all_states()
        st.rerun()

col_left, col_spacer, col_right = st.columns([1, 0.15, 3])

with col_left:
    with st.container():
        st.subheader("🥕 재료 입력")
        if st.session_state.uploaded_image:
            buf = BytesIO()
            st.session_state.uploaded_image.save(buf, format="PNG")
            img_str = base64.b64encode(buf.getvalue()).decode()
            st.markdown(f"""
            <div class="ingredient-circle-with-image">
                <img src="data:image/png;base64,{img_str}" class="ingredient-image">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ingredient-circle-empty">
                재료를 입력하고<br>요리를 추천받으세요!
            </div>
            """, unsafe_allow_html=True)

        uploaded = st.file_uploader("재료 사진 업로드", type=["jpg", "jpeg", "png"])
        if uploaded and not st.session_state.image_uploaded:
            image = Image.open(uploaded)
            st.session_state.uploaded_image = image
            st.session_state.image_uploaded = True
            st.rerun()

        with st.expander("🍽️ 희망 식단"):
            diet_col1, diet_col2 = st.columns(2)
            with diet_col1:
                st.checkbox("다이어트", key="diet_0")
                st.checkbox("저염", key="diet_2")
            with diet_col2:
                st.checkbox("채식", key="diet_1")
                st.checkbox("저탄고지", key="diet_3")
        with st.expander("⏰ 희망 조리시간"):
            time_col1, time_col2 = st.columns(2)
            with time_col1:
                st.checkbox("간단", key="time_0")
                st.checkbox("정성", key="time_2")
            with time_col2:
                st.checkbox("보통", key="time_1")
                st.checkbox("상관없음", key="time_3")
        with st.expander("📊 희망 난이도"):
            diff_col1, diff_col2 = st.columns(2)
            with diff_col1:
                st.checkbox("쉬움", key="diff_0")
                st.checkbox("어려움", key="diff_2")
            with diff_col2:
                st.checkbox("보통", key="diff_1")
                st.checkbox("상관없음", key="diff_3")

        if 'ingredient_input_value' not in st.session_state:
            st.session_state.ingredient_input_value = ""

        ingredients = st.text_input("예: 계란, 밥, 양파", 
                                   value=st.session_state.ingredient_input_value,
                                   key="ingredient_input")
        add_col, reset_col = st.columns(2)
        with add_col:
            if st.button("➕ 추가"):
                add_ingredients_without_duplicates(ingredients)
                st.session_state.ingredient_input_value = ""
                st.rerun()
        with reset_col:
            if st.button("♻️ 초기화"):
                st.session_state.ingredients_list = []
                st.session_state.ingredient_input_value = ""
                st.session_state.last_message = "✅ 모든 재료가 초기화되었습니다."
                st.session_state.message_type = "success"
                st.rerun()

        if st.session_state.last_message:
            if st.session_state.message_type == "success":
                st.success(st.session_state.last_message)
            elif st.session_state.message_type == "warning":
                st.warning(st.session_state.last_message)
            elif st.session_state.message_type == "info":
                st.info(st.session_state.last_message)
            elif st.session_state.message_type == "mixed":
                lines = st.session_state.last_message.split('\n')
                if len(lines) >= 2:
                    st.success(lines[0])
                    st.warning(lines[1])
                else:
                    st.info(st.session_state.last_message)
            if st.button("✖️ 메시지 지우기", key="clear_message"):
                st.session_state.last_message = None
                st.session_state.message_type = None
                st.rerun()

        st.markdown("**저장된 재료**")
        if st.session_state.ingredients_list:
            st.markdown("".join([f"<span class='ingredient-tag'>{i}</span>" for i in st.session_state.ingredients_list]), unsafe_allow_html=True)
        else:
            st.markdown("*아직 추가된 재료가 없습니다.*")

        st.session_state.user_input = get_user_input()
    if st.button("🔮 재료 분석하기"):
        ingredients = get_ingredients_input()
        preferences = get_preferences_input()
        if ingredients:
            st.session_state.ingredients_data = ingredients
            st.session_state.preferences_data = preferences
            start_analyzing()
            st.rerun()
        else:
            st.warning("재료를 먼저 입력해주세요")

    if 'ingredients_data' in st.session_state and 'preferences_data' in st.session_state:
        st.markdown("### 📦 input_module.py 함수 결과 미리보기")
        st.json({
            "ingredients": st.session_state.ingredients_data,
            "preferences": st.session_state.preferences_data
        })
recipe_instructions = {
    "오므라이스": """
**재료**
- 계란 1개
- 밥 1공기
- 양파 1/4개
- 당근 1/3개

**조리법**
1. 양파와 당근을 잘게 썰어 팬에 볶습니다.
2. 밥을 넣고 함께 볶아 간을 합니다.
3. 계란을 풀어 팬에 얇게 부칩니다.
4. 볶음밥을 계란으로 감싸면 완성!
""",
    "샐러드": """
**재료**
- 양상추
- 오이
- 토마토
- 드레싱

**조리법**
1. 야채를 깨끗이 씻고 물기를 제거합니다.
2. 먹기 좋은 크기로 썰어 그릇에 담습니다.
3. 원하는 드레싱을 뿌려 마무리합니다.
""",
    "토마토 파스타": """
**재료**
- 파스타면
- 토마토
- 양파
- 마늘

**조리법**
1. 파스타면을 끓는 물에 삶아줍니다.
2. 마늘, 양파를 볶다가 토마토를 넣어 소스를 만듭니다.
3. 삶은 면을 넣고 함께 볶습니다.
4. 소금, 후추로 간을 맞추면 완성!
"""
}

with col_right:
    if st.session_state.analyzing_mode:
        # 추천 레시피 바 대신 심플 텍스트 바
        st.markdown("""
        <div style="width:100%;padding:18px 0 18px 0;text-align:center;font-size:1.5em;font-weight:bold;border-bottom:1px solid #eee;background:rgba(255,255,255,0.7);">
        추천 레시피
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="recipe-content">', unsafe_allow_html=True)
        current_recipe = analyzed_recipes[st.session_state.current_recipe_index]
        recipe_col_left, recipe_col_right = st.columns([1, 1])
        with recipe_col_left:
            st.markdown(f"""
            <img src="{current_recipe['image']}"
                 style="width: 500px; display: block; margin: 70px auto 30px auto; border-radius: 20px;" />
            """, unsafe_allow_html=True)
        with recipe_col_right:
            st.markdown(f'<div class="analyzed-recipe-title">{current_recipe["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <span class="info-tag">⏱️ {current_recipe['cook_time']}</span>
                <span class="info-tag">📊 {current_recipe['difficulty']}</span>
                <span class="info-tag">🔥 {current_recipe['calories']}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="section-title">🥘 필요한 재료</div>', unsafe_allow_html=True)
            ingredients_html = "".join([f'<span class="ingredient-item">{ingredient}</span>' for ingredient in current_recipe['ingredients']])
            st.markdown(f'<div style="margin-bottom: 15px;">{ingredients_html}</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">👩‍🍳 조리법</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            for i, step in enumerate(current_recipe['instructions']):
                st.markdown(f"**{i+1}.** {step}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="decision-buttons">', unsafe_allow_html=True)
        st.markdown("### 이 레시피로 요리하시겠어요?")
        col_spacer, col_yes, col_no, col_spacer2 = st.columns([1, 2, 2, 1])
        with col_yes:
            if st.button("✅ Yes, 이걸로 할게요!", key="yes_btn"):
                handle_yes()
                st.rerun()
        with col_no:
            if st.button("❌ No, 다른 레시피 보여주세요", key="no_btn"):
                handle_no()
                st.rerun()
        st.markdown(f"**({st.session_state.current_recipe_index + 1} / {len(analyzed_recipes)})**")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif st.session_state.selected_recipe_mode:
        st.markdown("""
        <div style="width:100%;padding:18px 0 18px 0;text-align:center;font-size:1.5em;font-weight:bold;border-bottom:1px solid #eee;background:rgba(255,255,255,0.7);">
        선택된 레시피
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="recipe-content">', unsafe_allow_html=True)
        selected_recipe = analyzed_recipes[st.session_state.current_recipe_index]
        recipe_col_left, recipe_col_right = st.columns([1, 1])
        with recipe_col_left:
            st.markdown(f"""
            <img src="{selected_recipe['image']}"
                 style="width: 500px; display: block; margin: 70px auto 30px auto; border-radius: 20px;" />
            """, unsafe_allow_html=True)
        with recipe_col_right:
            st.markdown(f'<div class="analyzed-recipe-title">선택된 레시피: {selected_recipe["title"]}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="margin: 15px 0;">
                <span class="info-tag">⏱️ {selected_recipe['cook_time']}</span>
                <span class="info-tag">📊 {selected_recipe['difficulty']}</span>
                <span class="info-tag">🔥 {selected_recipe['calories']}</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('<div class="section-title">🥘 필요한 재료</div>', unsafe_allow_html=True)
            ingredients_html = "".join([f'<span class="ingredient-item">{ingredient}</span>' for ingredient in selected_recipe['ingredients']])
            st.markdown(f'<div style="margin-bottom: 15px;">{ingredients_html}</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">👩‍🍳 조리법</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-container">', unsafe_allow_html=True)
            for i, step in enumerate(selected_recipe['instructions']):
                st.markdown(f"**{i+1}.** {step}")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="decision-buttons">', unsafe_allow_html=True)
        st.markdown("### 이 레시피로 요리하시겠어요?")
        col_spacer, col_start, col_find, col_spacer2 = st.columns([1, 2, 2, 1])
        with col_start:
            if st.button("🍳 요리 시작하기!", key="start_cooking_btn"):
                st.success("요리를 시작해보세요! 맛있게 드세요 😋")
        with col_find:
            if st.button("🔍 다른 레시피 찾기", key="find_other_btn"):
                st.session_state.selected_recipe_mode = False
                st.session_state.analyzing_mode = False
                st.rerun()
    else:
        st.markdown("### 🍲 오늘의 추천 레시피")
        recipe_data = [
            ("오므라이스", "계란 1개, 밥 1공기, 양파 1/4개, 당근 1/3개", "show_omurice_recipe"),
            ("샐러드", "양상추, 오이, 토마토, 드레싱", "show_salad_recipe"),
            ("토마토 파스타", "파스타면, 토마토, 양파, 마늘", "show_pasta_recipe")
        ]
        st.markdown('<div class="recipe-row">', unsafe_allow_html=True)
        cols = st.columns(3)
        for idx, (title, ing, key) in enumerate(recipe_data):
            with cols[idx]:
                if not st.session_state[key]:
                    st.markdown(f"""
                        <div class="recipe-card">
                            <img src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/dog.jpg-FYr8K5lDgiYhZVWUA1kfs3vf5fSfO3.jpeg" class="recipe-image">
                            <div class="recipe-title">{title}</div>
                            <div style='font-size:13px;'>{ing}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    if st.button(f"{title} 조리법 보기", key=f"btn_{key}"):
                        st.session_state[key] = True
                        st.rerun()
                else:
                    st.markdown(f"### 🌟 {title} 조리법")
                    st.markdown(recipe_instructions[title])
                    if st.button(f"{title} 카드로 돌아가기", key=f"back_{key}"):
                        st.session_state[key] = False
                        st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)