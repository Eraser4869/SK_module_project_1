import base64                               # 이미지 인코딩 및 디코딩
import requests                             # AI 사용
import os                                   # API 키 저장
import mimetypes                            # 이미지 파일 형식 통합
import tempfile                             # 임시 파일 생성(임시 이미지 파일 경로)
from dotenv import load_dotenv              # 키 불러오기
from ultralytics import YOLO                # YOLOv8 모델을 불러오기 위한 라이브러리
from typing import List, Dict, Union        # 타입 힌트를 위한 모듈
from openai import OpenAI                   # OpenAI 사용
from PIL import Image                       # Base64 --> PIL
from io import BytesIO



class IngredientDetector:
    ''' 
    생성자
    model_path: str = 모델 파일 경로, 문자열 타입 명시
    '''
    def __init__(self):
        # 예외처리
        load_dotenv()
        try:
            # YOLO 모델 로드
            self.model_path = os.getenv("MODEL_PATH")
            self.model = YOLO(self.model_path) # YOLO 모델을 로드
            self.class_names = self.model.names  # 클래스 ID와 이름 매핑 딕셔너리 (예: {0: 'egg', 1: 'tomato'})

            # OpenAI 클라이언트 준비
            OPEN_API_KEY = os.getenv('OPEN_API_KEY')
            self.openai_client = OpenAI(api_key=OPEN_API_KEY)
        except FileNotFoundError as e:
            print('모델 파일을 찾을 수 없습니다. 모델의 경로를 다시 확인해주세요.')
            self.model = None
            if not self.model_path:
                print("환경변수 MODELPATH가 설정되지 않았습니다. .env 파일을 확인해주세요.")
                return  []
        except OpenAI.error.OpenAIError as e:
            print('OpenAI API 키가 잘못되었거나 설정되지 않았습니다. .env 파일을 확인해주세요.')
            self.openai_client = None
    
    # 식재료 분류 메서드
    def classify_ingredients(self, image_dicts: List[dict]) -> List[Dict[str, int]]:
        """
        image_path: base64로 변환된 이미지를 받음
        이미지에 포함된 식재료 리스트 반환
        param image_dicts: base64로 인코딩된 이미지 딕셔너리 리스트
        return: 각 이미지에 대한 식재료 리스트를 포함하는 리스트
        """
        
        # 모든 이미지의 감지 결과를 저장할 리스트
        all_labels = []
        # 이미지를 전달하기 위한 검사
        for i, image_dict in enumerate(image_dicts):
            try:
                img = self.dict_to_pil_image(image_dict)
                if img is None:
                    all_labels.append([])
                    continue
                # PIL --> 임시 파일 경로
                temp_path = self.pil_to_temp_path(img)

                # YOLO 모델 추론
                result = self.model(temp_path)[0]  # YOLO 감지 결과
                labels = [] # 현재 이미지의 식재료를 저장하는 리스트
                confidence_threshold = 0.8  # 이 값 이하이면 신뢰도가 낮다고 판단
                low_confidence_detected = False

                # YOLO가 탐지에 실패했을 시에 AI 사용
                if result.boxes is None or len(result.boxes) == 0:
                    #print(f'{i}번째 이미지에서는 객체를 감지하지 못했습니다. AI를 사용합니다.')
                    fallback_labels = self.detect_with_ai(temp_path)

                    # 중복 처리, all_labels와 비교 후 새로운 값만 추가
                    flattened=[]
                    for item in fallback_labels:
                        if isinstance(item, list):
                            flattened.extend(item)
                        else:
                            flattened.append(item)
                    # all_labels에 존재하는 전체 라벨 flatten
                    existing_labels = set(label for sublist in all_labels for label in sublist) # all_labels의 리스트 요소의 요소들을 추가
                    # 중복 제거 및 새 항목 추가
                    new_labels = [label for label in set(flattened) if label not in existing_labels]
                    all_labels.append(new_labels if new_labels else []) # new_labels가 비어 있지 않으면 추가, 비어있으면 [] 추가
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
                # label이 있을 시
                if labels:
                    all_labels.append(list(set(labels)))    # 현재 이미지 결과를 전체 리스트에 추가(중복 제거)
                # 신뢰도가 낮거나 객체 미탐지 시
                elif low_confidence_detected or result.boxes: 
                    fallback_labels = self.detect_with_ai(temp_path)
                    # 위와 같은 방식으로 중복 제거
                    flattened = []
                    for item in fallback_labels:
                        if isinstance(item, list):
                            flattened.extend(item)  # 리스트 안의 요소들을 꺼내서 추가
                        else:
                            flattened.append(item)  # 문자열이면 그대로 추가
                    existing_labels = set(label for sublist in all_labels for label in sublist)
                    new_labels = [label for label in set(flattened) if label not in existing_labels]
                    all_labels.append(new_labels if new_labels else []) # new_labels가 비어 있지 않으면 추가, 비어있으면 [] 추가

                    # 안전장치 YOLO, GPT 둘 다 아무런 결과를 내지 못했을 때 공리스트 반환
                else:  
                    all_labels.append([])
                
            except Exception as e:
                print(f'!!!에러 발생!!! {i}번째 이미지를 처리 중 오류가 발생했습니다. 오류: {e}')
                all_labels.append([])   # 오류가 난 경우 빈 리스트를 추가 후 계속 진행
            #임시 파일 경로 삭제
            finally:
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f'에러 발생!! 임시 파일 삭제 실패: {e}')
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
                # base64로 인코딩
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
        
    # base64 --> PIL 변경
    def dict_to_pil_image(self, image_dict:dict) -> Image.Image:
        """
        base64로 인코딩된 이미지를 딕셔너리 형태로 받아서 PIL 이미지 객체로 변환하는 메서드
        param image_dict: base64로 인코딩된 이미지 딕셔너리
        return: PIL 이미지 객체
        """
        try:
            b64_data = image_dict.get("data")
            if not b64_data:    # 데이터 필드가 비었을 시 에러
                raise ValueError("data 필드가 비어 있습니다.")
            
            # data:image/png;base64,... 형태일 경우 MIME 프리픽스 제거
            if b64_data.startswith("data:image"):
                b64_data = b64_data.split(",")[1]
            img_data = base64.b64decode(b64_data)
            image = Image.open(BytesIO(img_data))
            image.load()    # 오류 방지. 명시적 이미지 로딩
            return image
        except Exception as e:
            print(f'[dict_to_pil_image] base64 디코딩 또는 이미지 로딩 오류: {e}')
            return None

    # 임시 경로 생성 메서드: dict 형대로 YOLO에 넣을 수 없기 때문에 임시 파일로 저장 
    def pil_to_temp_path(self, pil_img) -> str:
        """ 
        PIL 이미지를 임시 파일로 저장하고 경로를 반환하는 메서드
        param pil_img: PIL 이미지 객체
        return: 임시 파일 경로 문자열  
        """
        try:
            format = pil_img.format if pil_img.format else "PNG"    # 기본 값

            extension = f".{format.lower()}"

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
            pil_img.save(temp_file.name, format=format)
            return temp_file.name
        except Exception as e:
            print(f'[pil_to_temp_path] PIL 이미지를 임시 파일로 저장하는 중 오류 발생: {e}')
            return ""
    #리스트 변환 메서드
    def return_ingredient(self, image_dict: List[dict]) -> List[str]: 
        """
        이미지 딕셔너리 리스트를 받아서
        1. classify_ingredients 메서드를 호출하여 식재료를 감지하고
        2. 결과를 평탄화하여 중복 제거 후 리스트 형태로 반환
        3. 오류 발생 시 오류 메시지를 포함한 딕셔너리 형태로 반환
        """
        try:
            all_labels = self.classify_ingredients(image_dict) # 감지 결과 가져오기
            flattened = list(set(label for sublist in all_labels for label in sublist)) 
            return flattened   # 결과 리스트 형태로 반환
        
        except Exception as e:
            print(f'return_ingredient 오류 발생: {e}')
            return [{"error": str(e)}]  # 오류도 리스트 안에 딕셔너리로 반환
        
        
#main파일(디버깅, 사용 방식)
def pil_image_to_dict(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return {
        "mode": img.mode,
        "size": img.size,
        "data": img_str
    }

def main():
    #yolo_model_path = "SK_module_project_1/src/back2/runs/food_ingredient_fresh/weights/best.pt"   #실제로는 환경변수 사용(env파일 등)
    img = Image.open("-00007_jpeg_jpg.rf.603ade3dc1b62901a3ab6452b58e9291.jpg") # 테스트용 이미지 파일
    img_dict = pil_image_to_dict(img)   # PIL 이미지를 딕셔너리로 변환
    detector = IngredientDetector()     # 인스턴스 생성
    print(detector.return_ingredient([img_dict]))       # 이미지 딕셔너리 리스트를 전달하여 식재료 감지 결과 출력

if __name__=="__main__":
    main()
