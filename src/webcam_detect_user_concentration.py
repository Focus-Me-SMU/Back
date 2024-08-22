import cv2
from ultralytics import YOLO
import torch
import time

# 모델 로드
model = YOLO('pth파일')  # 파일 경로, yolo로 'Look_Forward', 'awake', 'drowsy', 'yelling' 클래스 학습
model.model.eval()

# 웹캠 열기
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

class_names = ['Look_Forward', 'awake', 'drowsy', 'yelling'] 
last_awake_time = time.time()
warning_time_threshold = 10  # threshold값 이상 awake 아닐 시, warning

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model(frame)

    awake_detected = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            conf = box.conf.item()  

            if cls < len(class_names):
                label = f"{class_names[cls]} {conf:.2f}"
                
                if class_names[cls] == 'awake':
                    awake_detected = True
                    last_awake_time = time.time()

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 현재 시간이 마지막 'awake' 감지 시간보다 경고 시간 임계값 이상 크면 경고
    if not awake_detected and (time.time() - last_awake_time > warning_time_threshold):
        cv2.putText(frame, 'WARNING: You have to concentrate!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow('YOLO Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
