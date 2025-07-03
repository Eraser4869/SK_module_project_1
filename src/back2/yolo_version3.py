#YOLOv8n 모델 학습
from ultralytics import YOLO
import torch

if __name__=="__main__":
    #모델 정의
    model = YOLO("C:/Users/choyk/Documents/GitHub/SK_module_project_1/src/back2/runs2/scratch/yolov8_scratch/weights/best.pt") 


    #모델 학습
    model.train(
        data="data3.yaml",
        epochs=30,                  # scratch 학습은 에포크를 늘리는 편이 좋음
        lr0=0.00005,                 # 학습률 설정. 전이학습은 낮게 설정
        patience=10,                # 20 epoch 동안 개선 없으면 조기 종료
        imgsz=640,
        batch=16,
        device="0",                 # GPU 사용
        project="runs3/scratch",    # 결과 저장
        name="yolov8_scratch",      # 실험별 세부 폴더
        exist_ok=True,               # 폴더 존재 시 에러 무시
        augment=True,                # 데이터 증강 
        freeze=12                   # 백본 고정으로 기존 특징 추출력 유지
    )


    #best.pt 
    #ultralytics v8는 .pt로 저장한 뒤 불러올 수 있다.
    best_pt = "runs3/scratch/yolov8_scratch/weights/best.pt"