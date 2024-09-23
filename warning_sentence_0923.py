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
yolo_model = YOLOModel(r"C:\Users\world\OneDrive\바탕 화면\서버연습용_0910\yolo_0919.pt")
eye_tracking_model = load_model(r"C:\Users\world\OneDrive\바탕 화면\서버연습용_0910\eye_tracking_model.pth", device)

# 카운트 변수 및 플래그 초기화
all_frame_count = 0
Look_Forward_count = 0
awake_count = 0
drowsy_count = 0
yelling_count = 0
sequence = []
sentence_count = 0
previous_sentence_count = 0  # 이전 sentence_count를 저장하기 위한 변수
frame_counter = 0
warning_frame_threshold = 3000
next_sent = False  # 'next' 신호가 전송되었는지 확인하는 플래그 변수

# 클릭 이벤트 처리 경로
@app.route('/upload-frame/click_event', methods=['POST'])
def reset_counts():
    global all_frame_count, Look_Forward_count, awake_count, drowsy_count, yelling_count, sentence_count, next_sent
    app.logger.info("Click event received. Resetting all counts.")
    
    # 모든 카운트를 0으로 초기화
    all_frame_count = 0
    Look_Forward_count = 0
    awake_count = 0
    drowsy_count = 0
    yelling_count = 0
    sentence_count = 0  # sentence_count 초기화
    next_sent = False  # 'next' 신호 플래그 리셋

    return jsonify({'message': 'Counts reset successfully'}), 200

#next클릭하면 결과처리
@app.route('/upload-frame/click_next', methods=['POST'])
def reset_sentence_count():
    global sentence_count, next_sent
    app.logger.info("Click event received for 'next'. Resetting sentence_count and next_sent.")

    # sentence_count와 next_sent를 초기화
    sentence_count = 0
    next_sent = False

    return jsonify({'message': 'sentence_count and next_sent reset successfully'}), 200

#종료버튼 클릭하면 처리
@app.route('/upload-frame/click_end', methods=['POST'])
def return_counts():
    global all_frame_count, Look_Forward_count, awake_count, drowsy_count, yelling_count
    app.logger.info("Click event received for 'end'. Returning all counts.")

    counts = {
        'all_frame_count': all_frame_count,
        'Look_Forward_count': Look_Forward_count,
        'awake_count': awake_count,
        'drowsy_count': drowsy_count,
        'yelling_count': yelling_count
    }

    return jsonify(counts), 200


@app.route('/upload-frame', methods=['POST'])
def upload_frame():
    global frame_counter, sequence, sentence_count, previous_sentence_count, next_sent
    global all_frame_count, Look_Forward_count, awake_count, drowsy_count, yelling_count

    app.logger.debug("Received request")
    if 'frame' not in request.files:
        app.logger.error("No frame part in request")
        return jsonify({'error': 'No frame part'}), 400
    
    frame_file = request.files['frame']
    frame_bytes = frame_file.read()

    app.logger.debug(f"Frame bytes length: {len(frame_bytes)}")

    try:
        # 이미지 디코딩
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        app.logger.debug(f"OpenCV decoded frame shape: {img_cv2.shape if img_cv2 is not None else 'None'}")

        if img_cv2 is None:
            raise ValueError("OpenCV failed to decode the image")

        # YOLO 모델로 사용자 상태 분석
        results = yolo_model.predict(img_cv2)
        highest_confidence = -1  # 가장 높은 확률을 추적하는 변수
        predicted_class = None   # 가장 높은 확률을 가진 예측된 클래스
        awake_detected = False

        # 모든 예측 결과 중 가장 높은 확률의 클래스만 선택
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = box.conf.item()

                if cls < len(yolo_model.class_names):
                    # 가장 높은 확률을 가진 클래스를 선택
                    if conf > highest_confidence:
                        highest_confidence = conf
                        predicted_class = yolo_model.class_names[cls]
                        if predicted_class == 'awake':
                            awake_detected = True
                            frame_counter = 0  # 'awake' 탐지 시 프레임 카운터 초기화

        # 프레임이 처리되었으므로 전체 프레임 카운트를 증가
        all_frame_count += 1

        # 가장 높은 확률의 클래스에 따라 해당 상태의 카운트를 증가
        if predicted_class == 'Look_Forward':
            Look_Forward_count += 1
        elif predicted_class == 'awake':
            awake_count += 1
        elif predicted_class == 'drowsy':
            drowsy_count += 1
        elif predicted_class == 'yelling':
            yelling_count += 1

        # 가장 높은 확률의 클래스만 로그로 출력
        if predicted_class:
            app.logger.debug(f"Predicted class: {predicted_class} with confidence: {highest_confidence:.2f}")

        # 프레임 카운팅 (사용자가 'awake' 상태가 아니면 카운트 증가)
        if not awake_detected:
            frame_counter += 1
            app.logger.debug(f"'awake' not detected, frame count: {frame_counter}")

        # 경고 프레임 수 임계값을 넘었을 경우 경고 메시지 생성
        if frame_counter >= warning_frame_threshold:
            app.logger.warning("Warning: User not detected as 'awake' for 5 frames")
            frame_counter = 0
            return jsonify({
                'message': 'Warning: User not detected as awake for 5 frames',
                'sentence_count': sentence_count,
                'show_toast': True
            }), 200

        # ResNet 기반 눈 추적 모델로 사용자의 시선 방향 분석
        input_image = preprocess_image(img_cv2, device)
        eye_region = eye_tracking_model(input_image)
        eye_region = torch.argmax(eye_region).item()  # 0, 1, 2, 3 중 하나의 값

        app.logger.debug(f"Eye tracking region: {eye_region}")

        # 시선 추적 순서 확인 및 처리
        if eye_region == 0 and (len(sequence) == 0 or sequence[-1] == 3):
            sequence = [0]
        elif eye_region == 1 and sequence == [0]:
            sequence.append(1)
        elif eye_region == 2 and sequence == [0, 1]:
            sequence.append(2)
        elif eye_region == 3 and sequence == [0, 1, 2]:
            sequence.append(3)
            sentence_count += 1
            app.logger.info(f"Correct sequence detected. Sentence count increased: {sentence_count}")
            sequence = []  # 시퀀스를 초기화하여 다시 0부터 시작하도록 설정

        # sentence_count가 변경되었는지 확인
        show_toast = sentence_count != previous_sentence_count
        if show_toast:
            app.logger.info(f"Sentence count changed from {previous_sentence_count} to {sentence_count}")
            previous_sentence_count = sentence_count

        # 'next' 신호를 단 한 번만 반환할 조건 추가 (sentence_count가 5에 도달하고 아직 'next' 신호가 전송되지 않았을 경우)
        if sentence_count == 2 and not next_sent:
            app.logger.info("Sentence count reached 5 for the first time. Returning 'next' signal.")
            return jsonify({
                'message': 'next',
                'sentence_count': sentence_count,
                'show_toast': True
            }), 200

        # 응답 생성
        response = {
            'message': 'Frame processed successfully', 
            'eye_region': eye_region, 
            'sentence_count': sentence_count,
            'show_toast': show_toast,
            'all_frame_count': all_frame_count,
            'Look_Forward_count': Look_Forward_count,
            'awake_count': awake_count,
            'drowsy_count': drowsy_count,
            'yelling_count': yelling_count
        }

        if show_toast:
            response['toast_message'] = f"Sentence count: {sentence_count}"

        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error processing frame: {str(e)}")
        return jsonify({
            'error': f'Error processing frame: {str(e)}',
            'sentence_count': sentence_count,
            'show_toast': False
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
