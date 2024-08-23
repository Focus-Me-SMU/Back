from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import torch
import cv2
import numpy as np
from model import load_model, YOLOModel
from predict import predict_gaze
import time

app = FastAPI()

yolo_model_path = 'best.pt'
resnet_model_path = 'eye_tracking_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

yolo_model = YOLOModel(yolo_model_path)
resnet_model = load_model(resnet_model_path, device)

@app.post("/predict")
async def predict(frame: UploadFile = File(...), num_lines: int = Form(...)):
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    expected_labels = [0, 1, 2, 3]
    current_index = 0
    cycle_count = 0
    max_cycles = num_lines

    cycle_times = []

    confidence_thresholds = {
        'Look_Forward': 0.9,
        'awake': 0.4,
        'drowsy': 0.8,
        'yelling': 0.8
    }

    cycle_start_time = None

    yolo_results = yolo_model.predict(frame)
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            conf = box.conf.item()

            class_name = yolo_model.class_names[cls]

            if class_name in confidence_thresholds and conf >= confidence_thresholds[class_name]:
                if class_name == 'awake':
                    cropped_frame = frame[y1:y2, x1:x2]
                    predicted_label = predict_gaze(frame, resnet_model, device)
                    
                    if predicted_label[0] == expected_labels[current_index]:
                        if current_index == 0:
                            cycle_start_time = time.time()

                        current_index += 1

                        if current_index == len(expected_labels):
                            cycle_count += 1
                            current_index = 0
                            cycle_end_time = time.time()
                            cycle_duration = cycle_end_time - cycle_start_time
                            cycle_times.append(cycle_duration)

                            if cycle_count == max_cycles:
                                average_cycle_time = sum(cycle_times) / len(cycle_times)
                                return JSONResponse(content={
                                    "status": "completed",
                                    "line_times": cycle_times,
                                    "average_time": average_cycle_time
                                })

    return JSONResponse(content={"status": "in_progress", "line_times": cycle_times})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)