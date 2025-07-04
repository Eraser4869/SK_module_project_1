class SimpleRecipeClassifier:
    def __init__(self, recipe_dict):
        self.recipe_dict = recipe_dict

    def recommend(self, user_ingredients):
        result_dict = {}
        for rid, info in self.recipe_dict.items():
            재료_이름들 = [r["item"] for r in info["재료"]]
            점수 = sum(1 for ingr in user_ingredients if ingr in 재료_이름들)
            if 점수 > 0:
                result_dict[rid] = {
                    "재료": info["재료"],
                    "조미료": info["조미료"]
                }
        return result_dict
