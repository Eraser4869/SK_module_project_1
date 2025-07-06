import requests
import csv

API_KEY = "8b19e7a99c1748bab05e"
BASE_URL = f"http://openapi.foodsafetykorea.go.kr/api/{API_KEY}/COOKRCP01/json"

def get_total_count():
    url = f"{BASE_URL}/1/1"
    res = requests.get(url)
    if res.status_code != 200:
        print("❌ total_count 조회 실패")
        return 0
    data = res.json()
    return int(data.get("COOKRCP01", {}).get("total_count", 0))

def clean_ingredients(text):
    text = text.replace("●", "").replace("\n", ",").replace("·", ",")
    return ' '.join(text.strip().split())

def get_recipes(start, end):
    url = f"{BASE_URL}/{start}/{end}"
    res = requests.get(url)
    if res.status_code != 200:
        return []
    data = res.json()
    return data.get("COOKRCP01", {}).get("row", [])

def download_all_recipes(filename="recipes.csv"):
    total = get_total_count()
    if total == 0:
        print("❌ 총 레시피 수가 0입니다.")
        return

    all_recipes = []
    step = 1000
    for start in range(1, total + 1, step):
        end = min(start + step - 1, total)
        print(f"▶ {start} ~ {end} 수집 중...")
        rows = get_recipes(start, end)
        for row in rows:
            name = row.get("RCP_NM", "").strip()
            ingredients_raw = row.get("RCP_PARTS_DTLS", "").strip()
            if name and ingredients_raw:
                ingredients = clean_ingredients(ingredients_raw)
                all_recipes.append({
                    "요리이름": name,
                    "재료": ingredients
                })

    with open(filename, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["요리이름", "재료"])
        writer.writeheader()
        writer.writerows(all_recipes)

    print(f"\n✅ 총 {len(all_recipes)}개 저장 완료 → {filename}")

if __name__ == "__main__":
    download_all_recipes()
