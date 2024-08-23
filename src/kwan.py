from flask import Flask, request, jsonify
import cv2
import numpy as np
import logging
from PIL import Image
import io

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

@app.route('/upload-frame', methods=['POST'])
def upload_frame():
    app.logger.debug("Received request")
    if 'frame' not in request.files:
        app.logger.error("No frame part in request")
        return jsonify({'error': 'No frame part'}), 400
    
    frame_file = request.files['frame']
    format = request.form.get('format')
    width = request.form.get('width')
    height = request.form.get('height')
    
    app.logger.debug(f"Received frame - Format: {format}, Width: {width}, Height: {height}")

    frame_bytes = frame_file.read()
    app.logger.debug(f"Frame bytes length: {len(frame_bytes)}")

    try:
        # OpenCV를 사용한 디코딩
        nparr = np.frombuffer(frame_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        app.logger.debug(f"OpenCV decoded frame shape: {img_cv2.shape if img_cv2 is not None else 'None'}")

        if img_cv2 is None:
            raise ValueError("OpenCV failed to decode the image")

        # 이미지 저장
        cv2.imwrite('received_frame_cv2.jpg', img_cv2)
        
        # PIL을 사용한 디코딩 (추가 확인용)
        img_pil = Image.open(io.BytesIO(frame_bytes))
        app.logger.debug(f"PIL decoded image size: {img_pil.size}")
        
        img_pil.save('received_frame_pil.jpg')

        return jsonify({'message': 'Frame received and processed successfully'}), 200

    except Exception as e:
        app.logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'error': f'Error processing frame: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)