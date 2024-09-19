import torch
import time
from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import logging
from models import YOLOModel, load_model, preprocess_image  # Importing from models.py
from eyetracking_cycle import EyeTrackingCycle

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# YOLO 및 ResNet 모델 로드 (여기서는 가정된 모델 경로)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLOModel('yolo_user_concentration_detect.pt')
eye_tracking_model = load_model('eye_tracking_model.pth', device)

# 상태 변수
frame_counter = 0
non_awake_frame_counter = 0
warning_frame_threshold = 5
page_complete_threshold = 3  # 각 페이지당 문장 수
total_pages = 3  # 총 페이지 수
current_page = 1
current_sentence = 0
cycles_completed = 0
expected_eye_region_sequence = [0, 1, 2, 3]  # 정확한 시선 상태 순서
current_eye_sequence = []  # 현재 감지된 시선 상태 기록
cycle_start_time = None  # 사이클 시작 시간

# 시간 및 로그 저장 변수
program_start_time = time.time()
total_frame_count = 0
non_awake_frames = 0
cycle_times = []
page_data = {}

# 프로그램 시작 시 호출
def start_program():
    global program_start_time
    program_start_time = time.time()
    app.logger.info(f"Program started at {time.strftime('%H:%M:%S', time.localtime(program_start_time))}")

# 최종 보고서 생성
def generate_final_report():
    program_end_time = time.time()
    program_duration = program_end_time - program_start_time
    average_cycle_time = sum(cycle_times) / len(cycle_times) if cycle_times else 0
    return {
        'program_start_time': time.strftime('%H:%M:%S', time.localtime(program_start_time)),
        'program_end_time': time.strftime('%H:%M:%S', time.localtime(program_end_time)),
        'program_duration': program_duration,
        'total_frame_count': total_frame_count,
        'non_awake_frame_count': non_awake_frames,
        'cycle_times': cycle_times,
        'average_cycle_time': average_cycle_time,
        'page_data': page_data
    }

# 프레임 분석 및 결과 반환
@app.route('/upload-frame', methods=['POST'])
def upload_frame():
    global frame_counter, non_awake_frame_counter, total_frame_count, non_awake_frames, cycles_completed, current_page, current_sentence, current_eye_sequence, cycle_start_time

    if 'frame' not in request.files:
        app.logger.error("No frame part in request")
        return jsonify({'error': 'No frame part'}), 400
    
    frame_file = request.files['frame']
    frame_bytes = frame_file.read()

    try:
        # 이미지 디코딩
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_cv2 is None:
            raise ValueError("OpenCV failed to decode the image")

        # YOLO 모델로 사용자 상태 예측 (가정)
        results = yolo_model.predict(img_cv2)
        awake_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)
                if yolo_model.class_names[cls] == 'awake':
                    awake_detected = True

        # 총 프레임 수 카운트
        total_frame_count += 1

        # "awake" 상태 감지
        if awake_detected:
            non_awake_frame_counter = 0
        else:
            non_awake_frame_counter += 1
            non_awake_frames += 1

            if non_awake_frame_counter >= warning_frame_threshold:
                app.logger.warning("User has not been awake for more than 5 frames!")
                return jsonify({'warning': 'User has not been awake for more than 5 frames!'})

        # 시선 상태 처리
        eye_region = torch.argmax(eye_tracking_model(preprocess_image(img_cv2, device))).item()
        app.logger.debug(f"Current eye_region: {eye_region}")

        # 사이클 시작: eye_region == 0에서 시작 시간을 기록
        if len(current_eye_sequence) == 0 and eye_region == 0:
            # 사이클 시작 시간 기록
            cycle_start_time = time.time()
            current_eye_sequence.append(eye_region)
        elif len(current_eye_sequence) > 0 and eye_region == expected_eye_region_sequence[len(current_eye_sequence)]:
            current_eye_sequence.append(eye_region)

        # 사이클이 0 -> 1 -> 2 -> 3 순서로 완료되었을 때
        if current_eye_sequence == expected_eye_region_sequence and cycle_start_time is not None:
            cycle_end_time = time.time()
            cycle_duration = cycle_end_time - cycle_start_time  # 사이클의 총 지속 시간 계산
            cycle_times.append(cycle_duration)

            # 문장 읽기 완료 처리
            current_sentence += 1
            page_data[f'page{current_page}sentence{current_sentence}'] = {'duration': cycle_duration}

            # 시선 상태 초기화
            current_eye_sequence = []
            cycle_start_time = None  # 시작 시간을 초기화

            # 페이지 완료 처리
            if current_sentence >= page_complete_threshold:
                current_sentence = 0
                current_page += 1

                if current_page > total_pages:
                    return jsonify({'full_complete': True, 'end_button_enabled': True})
                else:
                    return jsonify({'page_complete': True, 'page': current_page - 1})

        return jsonify({'message': 'Frame processed successfully', 'eye_region': eye_region}), 200

    except Exception as e:
        app.logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500

# 프로그램 종료 후 최종 보고서 반환
@app.route('/final-report', methods=['POST'])
def final_report():
    return jsonify(generate_final_report()), 200

if __name__ == '__main__':
    start_program()
    app.run(host='0.0.0.0', port=5000, debug=True)