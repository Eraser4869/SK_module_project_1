import requests, json
from bs4 import BeautifulSoup

def food_info(name):
    url = f"https://www.10000recipe.com/recipe/list.html?q={name}"
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    else:
        print("HTTP response error:", response.status_code)
        return

    food_list = soup.find_all(attrs={'class': 'common_sp_link'})
    if not food_list:
        print("레시피를 찾을 수 없습니다.")
        return

    food_id = food_list[0]['href'].split('/')[-1]
    new_url = f'https://www.10000recipe.com/recipe/{food_id}'
    new_response = requests.get(new_url)
    if new_response.status_code == 200:
        html = new_response.text
        soup = BeautifulSoup(html, 'html.parser')
    else:
        print("HTTP response error:", new_response.status_code)
        return

    food_info = soup.find(attrs={'type': 'application/ld+json'})
    result = json.loads(food_info.text)

    ingredient = ', '.join(result['recipeIngredient'])
    recipe = [f"{i+1}. {step['text']}" for i, step in enumerate(result['recipeInstructions'])]

    res = {
        'name': name,
        'ingredients': ingredient,
        'recipe': recipe
    }

    return res

# 사용자 입력 받기
if __name__ == "__main__":
    food_name = input("레시피를 알고 싶은 음식 이름을 입력하세요: ")
    info = food_info(food_name)

    if info:
        print(f"\n음식 이름: {info['name']}")
        print(f"\n재료:\n{info['ingredients']}")
        print(f"\n레시피:")
        for step in info['recipe']:
            print(step)
