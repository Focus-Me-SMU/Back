import cv2
import logging
import time
import pygame
from ultralytics import YOLO
from predict import predict_gaze
from model import YOLOModel

# pygame 초기화
pygame.mixer.init()
warning_sound = pygame.mixer.Sound("카리나_일어나.mp3")  # "warning.mp3" 파일을 경고음으로 사용

# YOLO 로그 출력을 억제
logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

def process_webcam_stream(yolo_model, resnet_model, device):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    window_name = 'Gaze Tracking'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    expected_labels = [0, 1, 2, 3]
    current_index = 0
    cycle_count = 0
    max_cycles = 3

    # 클래스별 confidence threshold 설정
    confidence_thresholds = {
        'Look_Forward': 0.9,
        'awake': 0.4,
        'drowsy': 0.8,
        'yelling': 0.8
    }

    awake_start_time = time.time()
    warning_played = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        yolo_results = yolo_model.predict(frame)
        awake_detected = False
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = box.conf.item()

                class_name = yolo_model.class_names[cls]
                # print(f"Detected: {class_name} with confidence {conf}")

                if class_name in confidence_thresholds and conf >= confidence_thresholds[class_name]:
                    label = f"{class_name} {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                    if class_name == 'awake':
                        awake_detected = True
                        cropped_frame = frame[y1:y2, x1:x2]
                        predicted_label = predict_gaze(frame, resnet_model, device)
                        # print(f"Predicted Label: {predicted_label}")
                        resnet_label = f"Status: {predicted_label[0]}"
                        cv2.putText(frame, resnet_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                        # Check if the predicted label matches the expected label
                        if predicted_label[0] == expected_labels[current_index]:
                            current_index += 1
                            print(f"Label {predicted_label[0]} detected. Moving to next expected label.")

                            if current_index == len(expected_labels):
                                cycle_count += 1
                                current_index = 0
                                print(f"Cycle {cycle_count} complete.")

                                if cycle_count == max_cycles:
                                    print("Complete")
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    return

        if not awake_detected:
            if time.time() - awake_start_time > 10:
                cv2.putText(frame, "Warning: Please stay awake!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                if not warning_played:
                    warning_sound.play()
                    warning_played = True
        else:
            awake_start_time = time.time()
            warning_played = False

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 예시 모델과 장치를 사용하여 호출 (예를 들어, yolo_model과 resnet_model, device가 정의된 경우)
# yolo_model = YOLOModel(...)
# resnet_model = ...
# device = ...
# process_webcam_stream(yolo_model, resnet_model, device)
