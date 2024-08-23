import torch
import torchvision.models as models
import torch.nn as nn
from ultralytics import YOLO
import cv2
import torchvision.transforms as transforms
from PIL import Image as Image_pil

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.model.model.eval()
        self.class_names = ['Look_Forward', 'awake', 'drowsy', 'yelling']

    def predict(self, frame):
        results = self.model(frame)
        return results

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        n_features = self.model.fc.in_features
        self.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 4)
        )
        self.model.fc = self.fc

    def forward(self, x):
        x = self.model(x)
        return x

def load_model(model_path, device):
    model = ResNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image_pil.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    input_image = transform(image).unsqueeze(0).to(device)
    return input_image