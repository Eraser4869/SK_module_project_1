#YOLOv8n 모델 학습
from ultralytics import YOLO
import torch

if __name__=="__main__":
    #모델 정의
    model = YOLO("yolov8n.yaml") 


    #모델 학습
    model.train(
        data="data.yaml",
        epochs=100,            # scratch 학습은 에포크를 늘리는 편이 좋음
        imgsz=640,
        batch=16,
        device="0", #GPU 사용
        project="runs/scratch", #결과 저장
        name="yolov8_scratch",  #실험별 세부 폴더
        exist_ok=True           #폴더 존재 시 에러 무시
    )


    #best.pt 
    #ultralytics v8는 .pt로 저장한 뒤 불러올 수 있다.
    best_pt = "runs/scratch/yolov8_scratch/weights/best.pt"
    #model_scratch = YOLO(best_pt).model  # nn.Module 추출


