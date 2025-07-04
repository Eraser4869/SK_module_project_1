import streamlit as st

def get_ingredients_input():
    text_ingredients = st.session_state.get("ingredients_list", [])
    image_ingredients = []
    if st.session_state.get("uploaded_image"):
        image_ingredients.append(st.session_state["uploaded_image"])
    if not text_ingredients and not image_ingredients:
        st.warning("텍스트 또는 이미지 재료 중 하나는 반드시 있어야 합니다.")
        return None
    return {
        "text": text_ingredients,
        "image": image_ingredients
    }

def get_preferences_input():
    diet_choices = ["다이어트", "채식", "저염", "저탄고지"]
    time_choices = ["간단", "보통", "정성", "상관없음"]
    difficulty_choices = ["쉬움", "보통", "어려움", "상관없음"]

    # 각 항목별로 True/False 리스트로 반환
    diet_bools = [st.session_state.get(f"diet_{i}", False) for i in range(len(diet_choices))]
    time_bools = [st.session_state.get(f"time_{i}", False) for i in range(len(time_choices))]
    diff_bools = [st.session_state.get(f"diff_{i}", False) for i in range(len(difficulty_choices))]

    return {
        "diet": diet_bools,
        "time": time_bools,
        "difficulty": diff_bools
    }

# 사용 예시
ingredients = get_ingredients_input()
preferences = get_preferences_input()

if ingredients:  # 유효성 검사는 재료 쪽에서만
    # 백엔드에 전달
    # ex) requests.post(url, json={"ingredients": ingredients, "preferences": preferences})
    pass
