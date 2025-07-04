import streamlit as st
from io import BytesIO
import base64
import json

def pil_image_to_dict(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return {
        "mode": img.mode,
        "size": img.size,
        "data": img_str
    }

def get_ingredients_input():
    text_ingredients = st.session_state.get("ingredients_list", [])
    image_ingredients = []
    if st.session_state.get("uploaded_image"):
        # 이미지를 딕셔너리로 변환해서 추가
        image_ingredients.append(pil_image_to_dict(st.session_state["uploaded_image"]))
    if not text_ingredients and not image_ingredients:
        st.warning("텍스트 또는 이미지 재료 중 하나는 반드시 있어야 합니다.")
        return None
    return {
        "text": text_ingredients,
        "image": image_ingredients
    }

def get_preferences_input():
    diet_choices = ["다이어트", "채식", "저염", "저탄고지"]
    time_choices = ["간단", "보통", "정성"]
    difficulty_choices = ["쉬움", "보통", "어려움"]

    diet_bools = [1 if st.session_state.get(f"diet_{i}", False) else 0 for i in range(len(diet_choices))]
    time_bools = [1 if st.session_state.get(f"time_{i}", False) else 0 for i in range(len(time_choices))]
    diff_bools = [1 if st.session_state.get(f"diff_{i}", False) else 0 for i in range(len(difficulty_choices))]

    return {
        "diet": diet_bools,
        "time": time_bools,
        "difficulty": diff_bools
    }

# 사용 예시
ingredients = get_ingredients_input()
preferences = get_preferences_input()

if ingredients:
    # 딕셔너리 형태로 출력 (이미지 포함)
    st.write(json.dumps({"ingredients": ingredients, "preferences": preferences}, ensure_ascii=False))
    # ex) requests.post(url, json={"ingredients": ingredients, "preferences": preferences})
    pass
