from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv() 

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def extract_ingredients_with_gpt(user_input: str) -> list:
    prompt = f"""
    
- 식재료는 한국어 단어 형태로 리스트로 작성해 주세요.
- 문장 안의 불필요한 일반 단어, 조리 방법, 수량 표현, 장소, 사람 관련 단어는 제외해 주세요.
- 동의어가 있으면 모두 표준어(예: ‘달걀’→‘계란’)로 통일해 주세요.
- 명확히 식재료가 아닌 단어는 포함하지 말아 주세요.
- 결과는 Python 리스트 형태로 출력해 주세요.

입력: "{user_input}"
결과:
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 식재료 추출을 잘하는 AI야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content.strip()
        return eval(content) if content.startswith("[") else ["파싱 실패"]
    except Exception as e:
        print("[GPT 오류]", e)
        return []