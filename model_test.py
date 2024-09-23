import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models

# ResNet50 기반 모델 정의 (기존과 동일)
class ResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet, self).__init__()
        self.model = models.resnet50(weights=None)  # weights=None으로 설정 (사전학습 X)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 0, 1, 2 클래스 예측
        )

    def forward(self, x):
        return self.model(x)

# 모델 가중치 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(num_classes=3).to(device)
model.load_state_dict(torch.load('resnet_tablet.pt', map_location=device))
model.eval()

# OpenCV의 얼굴 검출기 사용
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50에 맞는 크기로 조정
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 웹캠에서 실시간 영상 처리
def predict_frame(frame):
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
    
    for (x, y, w, h) in faces:
        # 얼굴 영역 추출 및 크롭
        face_img = frame[y:y + h, x:x + w]
        
        # OpenCV 이미지를 PIL 이미지로 변환
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img_pil = Image.fromarray(face_img)
        
        # 전처리 적용
        input_tensor = transform(face_img_pil).unsqueeze(0).to(device)
        
        # 예측 수행
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = predicted.item()
        
        # 얼굴 주위에 초록색 사각형 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 예측 결과를 사각형 위에 텍스트로 표시
        label = f'Class: {predicted_class}'
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    return frame

# 웹캠 시작
cap = cv2.VideoCapture(0)  # 0번 웹캠(기본 웹캠) 사용

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break
    
    # 얼굴 검출 및 예측 결과 표시
    frame = predict_frame(frame)
    
    # 화면에 출력
    cv2.imshow('Webcam Face Detection', frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 창 닫기
cap.release()
cv2.destroyAllWindows()
