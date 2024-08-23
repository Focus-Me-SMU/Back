import torch
from model import load_model, YOLOModel
from webcam import process_webcam_stream

def main():
    yolo_model_path = 'best.pt'
    resnet_model_path = 'eye_tracking_model.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    yolo_model = YOLOModel(yolo_model_path)
    resnet_model = load_model(resnet_model_path, device)
    process_webcam_stream(yolo_model, resnet_model, device)

if __name__ == "__main__":
    main()
