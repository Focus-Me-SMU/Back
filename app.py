from flask import Flask, request, jsonify
import cv2
import numpy as np
import torch
import logging
from models import YOLOModel, load_model, preprocess_image  # 모듈에서 필요한 함수와 클래스를 가져옴

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# YOLO 및 ResNet 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLOModel(r'C:\Users\world\OneDrive\바탕 화면\졸프_0910\Back\src\yolo_user_concentration_detect.pt')
eye_tracking_model = load_model(r"C:\Users\world\OneDrive\바탕 화면\서버연습용_0910\eye_tracking_model.pth", device)

frame_counter = 0
warning_frame_threshold = 30

@app.route('/upload-frame', methods=['POST'])
def upload_frame():
    global frame_counter

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
        eye_region = torch.argmax(eye_region).item()

        app.logger.debug(f"Eye tracking region: {eye_region}")

        # 결과 반환
        return jsonify({'message': 'Frame processed successfully', 'eye_region': eye_region}), 200

    except Exception as e:
        app.logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
