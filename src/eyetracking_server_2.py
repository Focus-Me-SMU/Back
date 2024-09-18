
import torch
from flask import Flask, request, jsonify, Response
import cv2
import numpy as np
import logging
from models import YOLOModel, load_model, preprocess_image  # Importing from models.py
from eyetracking_cycle import EyeTrackingCycle

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# YOLO 및 ResNet 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLOModel('yolo_user_concentration_detect.pt')
eye_tracking_model = load_model('eye_tracking_resnet.pt', device)

# Flask 서버 관련 라우팅 등 기존 코드 유지...
