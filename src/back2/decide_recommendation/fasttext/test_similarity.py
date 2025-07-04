from gensim.models import FastText

# 모델 로드
model = FastText.load("ingredient_fasttext.model")

# 테스트할 단어
test_words = ["식초", "녹말", "녹말가루", "설탕", "소금", "마늘", "올리브오일", "후추"]

# 유사 단어 출력
for word in test_words:
    print(f"\n🔍 '{word}'와 유사한 단어 Top 5:")
    try:
        for similar_word, score in model.wv.most_similar(word, topn=5):
            print(f"  - {similar_word} ({score:.3f})")
    except KeyError:
        print("  - 단어가 사전에 없음")
