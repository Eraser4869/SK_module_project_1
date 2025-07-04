import json # JSON 형식으로 데이터를 직렬화하기 위해 사용
from ultralytics import YOLO # YOLOv8 모델을 불러오기 위한 라이브러리
from typing import List, Dict, Union # 타입 힌트를 위한 모듈

class IngredientDetector:
    ''' 
    생성자
    model_path: str = 모델 파일 경로, 문자열 타입 명시
    '''
    def __init__(self, model_path: str):
        #예외처리
        try:    
            self.model = YOLO(model_path) # YOLO 모델을 로드
            self.class_names = self.model.names  # 클래스 ID와 이름 매핑 딕셔너리 (예: {0: 'egg', 1: 'tomato'})
        except FileNotFoundError as e:
            print('파일을 찾을 수 없습니다. 모델의 경로를 다시 확인해주세요.')

    # 식재료 분류 메서드
    def classify_ingredients(self, image_paths: Union[str, List[str]]) -> List[Dict[str, int]]:
        """
        image_path: 하나 이상의 이미지 경로를 받음
        union[str, List[str]]: 문자열 하나 또는 문자열 리스트를 허용. 반환 타입 -> str
        반환값: [{'tomato': 2, 'egg': 1}, {'onion': 3}, ...] 등 각 이미지마다 감지된 식재료 이름과 개수를 담은 딕셔너리 리스트

        """
        if isinstance(image_paths, str):
            # 단일 문자열이면 리스트로 변환하여 일관된 처리
            image_paths = [image_paths]
            
        # 모든 이미지의 감지 결과를 저장할 리스트
        all_labels = []

        # 각 이미지에 대한 결과 반복
        for i, result in enumerate(self.model(image_paths)):    #오류 이미지의 순서를 알기 위해 추가
            try:
                labels = [] # 현재 이미지의 식재료를 저장하는 리스트
                if result.boxes is None or len(result.boxes) == 0:
                    print(f'{i}번째 이미지에서는 객체를 감지하지 못했습니다.')
                    all_labels.append([])  # 감지 실패 시 빈 리스트 추가
                    continue  # 다음 이미지로 이동
                for box in result.boxes:    # 감지된 객체들 반복
                    cls_id = int(box.cls)   # 클래스ID를 정수로 변환(0.0 --> 0 등)
                    label = self.class_names[cls_id]    # 클래스 ID를 이름으로 변환(1 --> 사과 등)
                    labels.append(label)    #이름을 리스트에 추가
                all_labels.append(list(set(labels)))   # 현재 이미지 결과를 전체 리스트에 추가(중복 제거)
            except Exception as e:
                print(f'!!!에러 발생!!! {i}번째 이미지를 처리 중 오류가 발생했습니다. 오류: {e}')
                all_labels.append([])   #오류가 난 경우 빈 리스트를 추가 후 계속 진행
                print(f'감지된 객체 없음. {i}번째 이미지')
                
        return all_labels
    
    #JSON 변환 메서드
    def to_json(self, image_input: Union[str, List[str], Dict[str, List[str]]]) -> str: 
        """
        이미지는 단일 객체, 리스트, 딕셔너리 형태로 반환
        classify_ingredients 결과를 JSON 형식으로 반환
        반환값: JSON 문자열
        """
        try:
            if isinstance(image_input,dict):    #입력이 딕셔너리면
                image_paths = image_input.get("이미지",[])  #이미지 키 로 값 꺼내오기. 없으면 공리스트 반환
            else:
                image_paths = image_input


            all_labels = self.classify_ingredients(image_paths) # 감지 결과 가져오기 
            return json.dumps({"results": all_labels}, ensure_ascii=False, indent=2)    # 한글은 없지만 한글이 안깨지게 설정. 들여쓰기까지(가독성)
        
        except Exception as e:
            print(f'to_json 오류 발생: {e}')
            return json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)
    

#main파일(디버깅용)

def main():
    #실제로는 환경변수 사용(env파일 등)
    yolo_model_path = "C:/Users/choyk/Documents/GitHub/SK_module_project_1/src/back2/runs/food_ingredient_fresh/weights/best.pt"
    #실제로는 받은 이미지 사용
    test_image_path = [""]
    detector = IngredientDetector(yolo_model_path)
    detector.to_json(test_image_path)




if __name__=="__main__":
    main()

