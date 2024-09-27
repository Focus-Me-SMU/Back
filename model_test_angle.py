import os
import cv2
import dlib
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
import torch.nn as nn

# 모델 정의: ResNet50 기반 모델
class ResNetWithAngle(nn.Module):
    def __init__(self):
        super(ResNetWithAngle, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)  # FC 레이어 변경
        
        # 각도 입력을 처리하는 레이어
        self.angle_fc = nn.Linear(3, 16)
        
        # 결합된 특징을 처리하는 최종 레이어
        self.combined_fc = nn.Linear(128 + 16, 3)  # 최종 분류 레이어 (3개의 클래스: 0, 1, 2)
    
    def forward(self, image, angle):
        image_features = self.resnet(image)
        angle_features = self.angle_fc(angle)
        combined_features = torch.cat((image_features, angle_features), dim=1)
        output = self.combined_fc(combined_features)
        return output

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 로드 및 평가 모드 설정
model = ResNetWithAngle().to(device)
model.load_state_dict(torch.load('resnet_tablet_angle.pt'))
model.eval()

# 얼굴 검출 모델 (dlib)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')  # dlib의 68 랜드마크 모델 로드

# 데이터 변환 (테스트 데이터에 적용)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean/std
])

# 얼굴 각도 계산 함수
def get_face_angle(shape):
    # 3D 모델 포인트 (정해진 얼굴 랜드마크의 3D 좌표)
    model_points = np.array([
        (0.0, 0.0, 0.0),    # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])
    
    # 랜드마크의 2D 이미지 좌표 추출
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),    # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye left corner
        (shape.part(45).x, shape.part(45).y),  # Right eye right corner
        (shape.part(48).x, shape.part(48).y),  # Left mouth corner
        (shape.part(54).x, shape.part(54).y)   # Right mouth corner
    ], dtype="double")

    # 카메라의 내부 매개변수 설정
    size = (1016, 762)
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # 왜곡 계수
    # solvePnP로 각도 계산
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)

    # 각도 벡터를 축으로 변환
    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    return angles[0], angles[1], angles[2]  # pitch, yaw, roll

# 웹캠 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1016)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 762)

frame_rate = 5  # 초당 5 프레임으로 제한
prev = 0

while True:
    time_elapsed = cv2.getTickCount() / cv2.getTickFrequency()
    ret, frame = cap.read()
    
    if not ret:
        break

    if time_elapsed > (1.0 / frame_rate):  # 초당 프레임 제한
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)  # 얼굴 검출
        
        for face in faces:
            shape = predictor(gray, face)  # 얼굴 랜드마크 검출

            # 얼굴 각도 계산
            pitch, yaw, roll = get_face_angle(shape)

            # 얼굴 크롭
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            face_image = frame[y:y+h, x:x+w]
            
            # 얼굴 이미지 크기 조정 및 전처리
            face_image = cv2.resize(face_image, (224, 224))
            face_tensor = transform(face_image).unsqueeze(0).to(device)
            angles_tensor = torch.tensor([[yaw, pitch, roll]], dtype=torch.float32).to(device)

            # 모델 예측
            with torch.no_grad():
                output = model(face_tensor, angles_tensor)
                _, predicted = torch.max(output, 1)
                label = predicted.item()

            # 결과 표시
            cv2.putText(frame, f'Label: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 이전 시간 갱신
        prev = cv2.getTickCount() / cv2.getTickFrequency()

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
