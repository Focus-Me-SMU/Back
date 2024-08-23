import cv2
import logging
import time
from ultralytics import YOLO
from predict import predict_gaze
from model import YOLOModel

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

    # 사이클 시간을 저장할 리스트
    cycle_times = []

    # 클래스별 confidence threshold 설정
    confidence_thresholds = {
        'Look_Forward': 0.9,
        'awake': 0.4,
        'drowsy': 0.8,
        'yelling': 0.8
    }

    cycle_start_time = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        yolo_results = yolo_model.predict(frame)
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
                        cropped_frame = frame[y1:y2, x1:x2]
                        predicted_label = predict_gaze(frame, resnet_model, device)
                        # print(f"Predicted Label: {predicted_label}")
                        resnet_label = f"Status: {predicted_label[0]}"
                        cv2.putText(frame, resnet_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

                        # Check if the predicted label matches the expected label
                        if predicted_label[0] == expected_labels[current_index]:
                            if current_index == 0:
                                # Start time for a new cycle
                                cycle_start_time = time.time()

                            current_index += 1
                            print(f"Label {predicted_label[0]} detected. Moving to next expected label.")

                            if current_index == len(expected_labels):
                                cycle_count += 1
                                current_index = 0
                                cycle_end_time = time.time()
                                cycle_duration = cycle_end_time - cycle_start_time
                                cycle_times.append(cycle_duration)
                                print(f"Cycle {cycle_count} complete. Duration: {cycle_duration:.2f} seconds.")

                                if cycle_count == max_cycles:
                                    average_cycle_time = sum(cycle_times) / len(cycle_times)
                                    print(f"Average cycle time: {average_cycle_time:.2f} seconds.")
                                    cap.release()
                                    cv2.destroyAllWindows()
                                    return

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()