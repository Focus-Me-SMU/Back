from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import logging
from ultralytics import YOLO

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)


model = YOLO(r'C:\Users\world\OneDrive\바탕 화면\졸프_0910\Back\src\yolo_user_concentration_detect.pt')  # YOLO 모델 경로
model.model.eval()
class_names = ['Look_Forward', 'awake', 'drowsy', 'yelling']

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
        
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        app.logger.debug(f"OpenCV decoded frame shape: {img_cv2.shape if img_cv2 is not None else 'None'}")

        if img_cv2 is None:
            raise ValueError("OpenCV failed to decode the image")

     
        results = model(img_cv2)
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
                        frame_counter = 0 

      
        if not awake_detected:
            frame_counter += 1
            app.logger.debug(f"'awake' not detected, frame count: {frame_counter}")

       
        if frame_counter >= warning_frame_threshold:
            app.logger.warning("User has not been awake for 30 frames!")
            frame_counter = 0  

        return jsonify({'message': 'Frame processed successfully'}), 200

    except Exception as e:
        app.logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
