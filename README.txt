<div align="center">

# 🧠 AI 기반 맞춤형 레시피 추천 시스템  
SK Module Project 1 - 4조
<img src="https://img.shields.io/badge/Python-3.9.13-blue?logo=python" />
<img src="https://img.shields.io/badge/GPT-4-enabled-brightgreen?logo=openai" />
<img src="https://img.shields.io/badge/License-Private-lightgrey" />

</div>

---

## 🥗 소개

> 사용자의 보유 재료와 식단/시간/난이도 선호도를 기반으로,  
> AI와 GPT-4 API를 통해 **맞춤형 레시피**를 추천하는 스마트 요리 추천 시스템입니다.

---

## ✨ 주요 기능

| 기능 구분 | 설명 |
|-----------|------|
| 🥄 재료 추출 | 문장 내 불필요한 요소를 제거하고 재료만 GPT로 추출 |
| 🧂 식단 분류기 | 영양 정보 기반의 다중 분류기와 간단한 룰 기반 분류기 지원 |
| ⏱️ 조리 시간 예측 | 레시피의 조리 난이도 및 시간을 ML 모델로 예측 |
| 🍽 맞춤형 추천 | 유저 선호도와 재료를 기반으로 상위 레시피 추천 |
| 🧠 GPT 보완 | 추천이 부족할 경우 GPT가 자동으로 요리를 생성 |

---

🧠 시스템 구조
📦 SK_module_project_1
├── 📁 data
├── 📄 crawling.py              # 레시피 크롤러
├── 📄 GPTAPI.py               # GPT 기반 재료 추출
├── 📄 ai_classifier_multi_agent.py
├── 📄 cooking_time_model.py
├── 📄 run_recommendation.py    # 메인 추천 스크립트
├── 📄 recipe_data_integrator.py
├── 📄 recipe_recommend.py
├── 📄 Interface.py             # 통합 실행용
├── 📄 nutrition_items.json / .pt / .pkl
└── ...

🧪 기술 스택
🐍 Python 3.9.13

🌈 Streamlit

🧠 OpenAI GPT-4 API

🧪 Scikit-Learn, Pandas

🌐 Foodsafety Korea OpenAPI

📊 Nutrition Embedding 기반 유사도 분석

<div align="center">
✨ 맛있고 건강한 AI 요리를 즐기세요! 🍳

</div> ```

