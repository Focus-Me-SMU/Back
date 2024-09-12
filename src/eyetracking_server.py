import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import logging
from ultralytics import YOLO
from PIL import Image as Image_pil
from torch import nn
import torchvision.models as models
import time

# eyetracking_cycle.py에서 EyeTrackingCycle 불러오기
from eyetracking_cycle import EyeTrackingCycle

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# YOLO 모델 클래스
class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.model.eval()
        self.class_names = ['Look_Forward', 'awake', 'drowsy', 'yelling']

    def predict(self, frame):
        results = self.model(frame)
        return results

# ResNet50 기반 눈 추적 모델 정의
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        n_features = self.model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)  # 0, 1, 2, 3 4개의 클래스를 위한 출력층
        )
        self.model.fc = self.fc

    def forward(self, x):
        x = self.model(x)
        return x

# 모델 로드 함수
def load_model(model_path, device):
    model = ResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# 이미지 전처리 함수
def preprocess_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image_pil.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_image = transform(image).unsqueeze(0).to(device)
    return input_image

# YOLO 및 ResNet 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLOModel('yolo_user_concentration_detect.pt')
eye_tracking_model = load_model('eye_tracking_model.pth', device)

# 시선 추적 사이클 관리 클래스 인스턴스 생성
# 문장의 수를 받아서 초기화
eye_cycle_tracker = EyeTrackingCycle(sentence_count=3)  # 예시로 3문장이 있다고 가정

frame_counter = 0
warning_frame_threshold = 30

@app.route('/upload-frame', methods=['POST'])
def upload_frame():
    global frame_counter

    #app.logger.debug("Received request")
    if 'frame' not in request.files:
        app.logger.error("No frame part in request")
        return jsonify({'error': 'No frame part'}), 400
    
    frame_file = request.files['frame']
    frame_bytes = frame_file.read()

    #app.logger.debug(f"Frame bytes length: {len(frame_bytes)}")

    try:
        # 이미지 디코딩
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        #app.logger.debug(f"OpenCV decoded frame shape: {img_cv2.shape if img_cv2 is not None else 'None'}")

        if img_cv2 is None:
            raise ValueError("OpenCV failed to decode the image")

        # YOLO 모델로 사용자 상태 분석
        results = yolo_model.predict(img_cv2)
        awake_detected = False

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = box.conf.item()

                if cls < len(yolo_model.class_names):
                    label = f"{yolo_model.class_names[cls]} {conf:.2f}"

                    if yolo_model.class_names[cls] == 'awake':
                        awake_detected = True
                        frame_counter = 0

        # 프레임 카운팅
        if not awake_detected:
            frame_counter += 1
            app.logger.debug(f"'awake' not detected, frame count: {frame_counter}")

        if frame_counter >= warning_frame_threshold:
            app.logger.warning("User has not been awake for 30 frames!")
            frame_counter = 0

        # ResNet 기반 눈 추적 모델로 사용자의 시선 방향 분석
        input_image = preprocess_image(img_cv2, device)
        eye_region = eye_tracking_model(input_image)
        eye_region = torch.argmax(eye_region).item()  # 0, 1, 2, 3 중 하나의 값

        app.logger.debug(f"Eye tracking region: {eye_region}")

        # 시선 추적 사이클 업데이트
        eye_cycle_tracker.update_state(eye_region)
        
        print(eye_region)

        # 모든 문장을 다 읽었는지 확인
        if eye_cycle_tracker.all_cycles_completed():
            # 프로그램 가동 시간, 평균 시간 계산
            program_runtime = eye_cycle_tracker.get_program_running_time()
            avg_time_per_sentence = eye_cycle_tracker.get_average_time_per_sentence()
            detailed_logs = eye_cycle_tracker.get_detailed_logs()

            # 응답 데이터 생성
            response_data = {
                'message': 'All sentences read successfully',
                'program_runtime': program_runtime,
                'average_cycle_time': avg_time_per_sentence,
                'detailed_logs': detailed_logs
            }

            # JSON 데이터를 서버에서 콘솔에 출력
            print(response_data)  # 콘솔에 출력

            # 결과 반환 (모든 문장을 다 읽었을 때만 반환)
            return jsonify(response_data), 200

        # 문장을 다 읽기 전에는 응답을 하지 않음
        return jsonify({'message': 'Still processing'}), 200

    except Exception as e:
        app.logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)