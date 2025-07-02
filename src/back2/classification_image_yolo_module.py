import json # JSON 형식으로 데이터를 직렬화하기 위해 사용
from ultralytics import YOLO # YOLOv8 모델을 불러오기 위한 라이브러리
from typing import List, Dict, Union # 타입 힌트를 위한 모듈

class IngredientDetector:
    ''' 
    생성자
    model_path: str = 모델 파일 경로, 문자열 타입 명시
    '''
    def __init__(self, model_path: str):
        self.model = YOLO(model_path) # YOLO 모델을 로드
        self.class_names = self.model.names  # 클래스 ID와 이름 매핑 딕셔너리 (예: {0: 'egg', 1: 'tomato'})

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
        # 모델을 통한 분류 실행
        results = self.model(image_paths)
        # 모든 이미지의 감지 결과를 저장할 리스트
        all_labels = []

        # 각 이미지에 대한 결과 반복
        for result in results:
            labels = [] # 현재 이미지의 식재료를 저장하는 리스트
            for box in result.boxes:    # 감지된 객체들 반복
                cls_id = int(box.cls)   # 클래스ID를 정수로 변환(0.0 --> 0 등)
                label = self.class_names[cls_id]    # 클래스 ID를 이름으로 변환(1 --> 사과 등)
                labels.append(label)    #이름을 리스트에 추가
            all_labels.append(list(set(labels)))   # 현재 이미지 결과를 전체 리스트에 추가(중복 제거)

        return all_labels
    #JSON 변환 메서드
    def to_json(self, image_paths: Union[str, List[str]]) -> str: 
        """
        classify_ingredients 결과를 JSON 형식으로 반환
        반환값: JSON 문자열
        """
        all_labels = self.classify_ingredients(image_paths) # 감지 결과 가져오기 
        return json.dumps({"results": all_labels}, ensure_ascii=False, indent=2)    # 한글은 없지만 한글이 안깨지게 설정. 들여쓰기까지(가독성)
    

#main파일

def main():
    yolo_model_path = "src/back2/runs/scratch/yolov8_scratch/weights/best.pt"   #실제로는 환경변수 사용(env파일)
    #실제로는 받은 이미지 사용
    test_image_path = [""]
    detector = IngredientDetector()




if __name__=="__main__":
    main()

