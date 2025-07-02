import re
from kiwipiepy import Kiwi
from typing import List

class IngredientExtractor:
    def __init__(self):
        self.kiwi = Kiwi()
        self.noun_tags = {"NNG", "NNP", "NNB"}
        self.ingredient_history = set()  # 누적된 재료 저장
        
        # 동의어 단어를 표준어로 매핑
        self.synonyms = {
            "달걀": "계란",
            "계란노른자": "계란",
            "계란흰자": "계란",
        }

    def fix_spacing(self, text: str) -> str:
        def join_if_spaced(word):

            # 한 글자씩 띄어쓰기 된 경우 붙여서 처리
            if re.match(r"^(?:[가-힣a-zA-Z]\s?)+$", word):
                return word.replace(" ", "")
            return word
        
        # 쉼표 기준으로 분리 후 각 조각 띄어쓰기 문제 수정
        parts = [join_if_spaced(part) for part in re.split(r"\s*,\s*", text)]
        return ", ".join(parts)

    def extract_and_add(self, text: str) -> List[str]:
        # 띄어쓰기 오류 수정
        fixed_text = self.fix_spacing(text)
        
        # 쉼표나 공백으로 분리
        chunks = re.split(r"\s*,\s*|\s+", fixed_text.strip())

        for chunk in chunks:
            if not chunk:
                continue
            # 형태소 분석해서 명사만 추출
            analyzed = self.kiwi.analyze(chunk)
            for word, tag, _, _ in analyzed[0][0]:
                if tag in self.noun_tags:
                    # 동의어가 있으면 표준어로 변환
                    standard_word = self.synonyms.get(word, word)
                    # 누적 집합에 추가
                    self.ingredient_history.add(standard_word)

        # 누적 재료를 정렬해서 반환
        return sorted(self.ingredient_history)

# 실행
if __name__ == "__main__":
    extractor = IngredientExtractor()

    while True:
        user_input = input("재료를 입력하세요. (종료하려면 'exit'): ").strip()
        if user_input.lower() == 'exit':
            break
        result = extractor.extract_and_add(user_input)
        print("현재까지 입력된 재료:", result)
