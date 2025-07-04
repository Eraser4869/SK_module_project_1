from ultralytics import YOLO
if __name__=="__main__":
    model = YOLO("C:/Users/choyk/Documents/GitHub/SK_module_project_1/src/back2/runs/scratch/yolov8_scratch/weights/best.pt") 


    model.train(
        data="data2.yaml",      # 새로운 데이터셋 경로
        epochs=150,              # 추가로 학습할 에폭 수
        imgsz=640,              # 이미지 크기
        batch=16,               # 배치 크기
        resume=False,           # resume이 아닌 fresh training으로 간주
        lr0=0.0015,           # (선택) 초기 학습률 - 필요 시 조정
        patience=10,             # 10번 학습 동안 개선이 없으면 조기 종료
        project="runs/scratch", #결과 저장
        name="yolov8_scratch",  #실험별 세부 폴더
        device=0,                # 0번 GPU 명시적으로 사용
        augment=True,            # 기본 augment 활성화
        mosaic=True,              # 모자이크 증강 적용
        exist_ok=True           #폴더 존재 시 에러 무시
    )


    best_pt = "runs/scratch/yolov8_scratch/weights/best.pt"


