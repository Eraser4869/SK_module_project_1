import json                                 # JSON 형식으로 데이터를 직렬화하기 위해 사용
import base64                               # 이미지 인코딩 및 디코딩
import requests                             # AI 사용
import os                                   # API 키 저장
import mimetypes                            # 이미지 파일 형식 통합
from dotenv import load_dotenv              # 키 불러오기
from ultralytics import YOLO                # YOLOv8 모델을 불러오기 위한 라이브러리
from typing import List, Dict, Union        # 타입 힌트를 위한 모듈
from openai import OpenAI                   # OpenAI 사용


class IngredientDetector:
    ''' 
    생성자
    model_path: str = 모델 파일 경로, 문자열 타입 명시
    '''
    def __init__(self, model_path: str):
        # 예외처리
        try:    
            self.model = YOLO(model_path) # YOLO 모델을 로드
            self.class_names = self.model.names  # 클래스 ID와 이름 매핑 딕셔너리 (예: {0: 'egg', 1: 'tomato'})
        except FileNotFoundError as e:
            print('모델 파일을 찾을 수 없습니다. 모델의 경로를 다시 확인해주세요.')
        # OpenAI 클라이언트 준비
        load_dotenv()
        OPEN_API_KEY = os.getenv('OPEN_API_KEY')
        self.openai_client = OpenAI(api_key=OPEN_API_KEY)

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
        for i, img_path in enumerate(image_paths):    # 이미지의 경로
            try:
                result = self.model(img_path)[0]  # YOLO 감지 결과

                labels = [] # 현재 이미지의 식재료를 저장하는 리스트

                confidence_threshold = 0.8  # 이 값 이하이면 신뢰도가 낮다고 판단
                low_confidence_detected = False

                # YOLO가 탐지에 실패했을 시에 AI 사용
                if result.boxes is None or len(result.boxes) == 0:
                    print(f'{i}번째 이미지에서는 객체를 감지하지 못했습니다. AI를 사용합니다.')
                    fallback_labels = self.detect_with_ai(img_path)
                    all_labels.append(fallback_labels)  # 감지 실패 시 빈 리스트 추가
                    continue  # 다음 이미지로 이동
                
                # YOLO의 신뢰도가 낮으면 AI 사용
                for box in result.boxes:    # 감지된 객체들 반복
                    conf = float(box.conf)
                    if conf >= confidence_threshold:
                        cls_id = int(box.cls)   # 클래스ID를 정수로 변환(0.0 --> 0 등)
                        label = self.class_names[cls_id]    # 클래스 ID를 이름으로 변환(1 --> 사과 등)
                        labels.append(label)    # 이름을 리스트에 추가
                    else:
                        low_confidence_detected = True # 신뢰도 낮은 객체 있음
                    
                    if labels:
                        all_labels.append(list(set(labels)))    # 현재 이미지 결과를 전체 리스트에 추가(중복 제거)

                    elif low_confidence_detected or result.boxes: # 신뢰도가 낮거나 객체 미탐지 시
                        fallback_labels = self.detect_with_ai(img_path)
                        all_labels.append(fallback_labels)
                    else:   # 안전장치 YOLO, GPT 둘 다 아무런 결과를 내지 못했을 때 공리스트 반환
                        all_labels.append([])
                
            except Exception as e:
                print(f'!!!에러 발생!!! {i}번째 이미지를 처리 중 오류가 발생했습니다. 오류: {e}')
                all_labels.append([])   # 오류가 난 경우 빈 리스트를 추가 후 계속 진행
        return all_labels
    
    # AI를 활용한 보조 탐지 메서드
    def detect_with_ai(self, image_path: str) -> List[str]:   # 이미지 경로를 리스트 형식으로 반환
        try:
            # 이미지 형식 확인
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = 'image/jpeg'    # 기본값 설정

            # 이미지를 base64로 인코딩
            with open(image_path, 'rb') as img_file:
                ''' 
                이미지를 바이너리 읽기 모드로 연 후 인코딩. --> GPT가 읽을 수 있는 형태로 변환하기 위해 
                utf-8로 디코딩하여 바이트 --> 문자열로 변환 
                '''
                img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
                

                # response 객체 생성
                response = self.openai_client.responses.create(
                    model='gpt-4.1',
                    input=[{
                        # 역할
                        "role": "developer",
                        "content": "식재료에 대해 잘 알고 있어."
                    },
                    {
                        # 질문
                        "role": "user",
                        "content": [
                            # 이미지는 실제 URL 없이 이미지 전달
                            {"type": "input_text", "text": "이 이미지에는 어떤 식재료들이 보이나요? 식재료의 이름들만 쉼표로 구분해서 알려주세요."},
                            {
                                "type": "input_image", 
                                "image_url":  f"data:{mime_type};base64,{img_base64}"
                            }
                        ]
                    }],
                    max_output_tokens=200   # 간단한 응답만 필요
                )
                # 결과 반환
                answer = response.output_text
                # 응답 결과 간단하게 파싱: 쉼표 기준 분리
                return [x.strip() for x in answer.split(',')]

        except Exception as e:
            print(f'GPT 문제 발생: {e}')
            return []

    #JSON 변환 메서드
    def to_json(self, images: Union[str, List[str], Dict[str, List[str]]]) -> str: 
        """
        이미지는 단일 객체, 리스트, 딕셔너리 형태로 반환
        classify_ingredients 결과를 JSON 형식으로 반환
        반환값: JSON 문자열
        """
        try:
            if isinstance(images,dict):    #입력이 딕셔너리면
                image_paths = images.get("이미지",[])  #이미지 키 로 값 꺼내오기. 없으면 공리스트 반환
            else:
                image_paths = images


            all_labels = self.classify_ingredients(image_paths) # 감지 결과 가져오기 
            return json.dumps({"results": all_labels}, ensure_ascii=False, indent=2)    # 한글은 없지만 한글이 안깨지게 설정. 들여쓰기까지(가독성)
        
        except Exception as e:
            print(f'to_json 오류 발생: {e}')
            return json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)
        
        
#main파일(디버깅용)

def main():
    yolo_model_path = "C:/Users/choyk/Documents/GitHub/SK_module_project_1/src/back2/runs/food_ingredient_fresh/weights/best.pt"   #실제로는 환경변수 사용(env파일)
    test_image_path = [".jpg"]
    detector = IngredientDetector(yolo_model_path)
    detector.to_json(test_image_path)
    #print(detector.to_json(test_image_path))



if __name__=="__main__":
    main()
