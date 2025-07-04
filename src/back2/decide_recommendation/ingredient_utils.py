
def tokenize_ingredient_string(raw: str) -> list[str]:
    """재료 문자열을 리스트로 변환"""
    return [i.strip() for i in raw.replace("\n", ",").split(",") if i.strip()]