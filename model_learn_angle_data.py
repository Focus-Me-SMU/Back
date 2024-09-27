import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet50_Weights

# 경로 설정
image_dir = 'eye_region_face'
csv_file = 'angle_data.csv'

# CSV 파일에서 각도 데이터를 불러오기
angle_data = pd.read_csv(csv_file)

# 디버깅용: CSV의 이미지 이름을 확인하여 제대로 불러왔는지 확인
print("CSV image names:", angle_data['image_name'].tolist()[:5])

# 이미지 및 각도 데이터를 로딩하는 PyTorch Dataset 클래스 정의
class ImageAngleDataset(Dataset):
    def __init__(self, image_dir, angle_data, transform=None):
        self.image_dir = image_dir
        self.angle_data = angle_data
        self.transform = transform
        self.image_paths = []
        self.angles = []
        self.labels = []
        
        # 데이터 로드
        for root, _, files in os.walk(self.image_dir):
            for file_name in files:
                # 증강된 이미지 파일 경로
                image_path = os.path.join(root, file_name)
                
                # 증강 이미지 이름에서 원본 이미지의 이름 추출 (예: 'eye_region_0_000_aug_0.jpg' -> 'eye_region_0_000')
                base_name = file_name.split('_aug')[0]  # 확장자를 떼고 '_aug' 이전의 부분만 남김
                
                # 각도 정보 매칭
                angle_row = angle_data[angle_data['image_name'] == base_name]

                if len(angle_row) > 0:
                    yaw, pitch, roll = angle_row[['yaw', 'pitch', 'roll']].values[0]
                    # 레이블 추출 (0, 1, 2)
                    label = int(base_name.split('_')[2])  # 'eye_region_0_000'에서 '0' 추출
                    
                    self.image_paths.append(image_path)
                    self.angles.append([yaw, pitch, roll])
                    self.labels.append(label)
                else:
                    # 매칭 실패 디버깅 출력
                    print(f"Image '{file_name}' with base '{base_name}' has no matching angle data in CSV.")

        # 로드된 이미지의 개수 확인
        print(f"Loaded {len(self.image_paths)} images from {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 이미지 로드 및 전처리
        image = cv2.imread(self.image_paths[idx])
        image = cv2.resize(image, (224, 224))
        if self.transform:
            image = self.transform(image)
        
        # 각도 및 레이블 가져오기
        angle = torch.tensor(self.angles[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, angle, label

# 데이터 변환 및 전처리 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean/std
])

# 모델 정의: ResNet50 기반 모델
class ResNetWithAngle(nn.Module):
    def __init__(self):
        super(ResNetWithAngle, self).__init__()
        # 기존 pretrained=True 대신 weights=ResNet50_Weights.DEFAULT 사용
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)  # FC 레이어 변경
        
        # 각도 입력을 처리하는 레이어
        self.angle_fc = nn.Linear(3, 16)
        
        # 결합된 특징을 처리하는 최종 레이어
        self.combined_fc = nn.Linear(128 + 16, 3)  # 최종 분류 레이어 (3개의 클래스: 0, 1, 2)
    
    def forward(self, image, angle):
        # ResNet50을 통해 이미지 특징 추출
        image_features = self.resnet(image)
        
        # 각도 특징 추출
        angle_features = self.angle_fc(angle)
        
        # 이미지 특징과 각도 특징 결합
        combined_features = torch.cat((image_features, angle_features), dim=1)
        
        # 최종 분류
        output = self.combined_fc(combined_features)
        
        return output

# 모델 학습 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20, model_path='resnet_tablet_angle.pt'):
    best_val_loss = float('inf')
    patience = 3
    trigger_times = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, angles, labels in train_loader:
            images, angles, labels = images.to(device), angles.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images, angles)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 검증 손실 계산
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, angles, labels in val_loader:
                images, angles, labels = images.to(device), angles.to(device), labels.to(device)
                outputs = model(images, angles)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # EarlyStopping 및 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f'Model saved at epoch {epoch+1} with val_loss: {val_loss:.4f}')
            trigger_times = 0
        else:
            trigger_times += 1
            print(f'EarlyStopping Trigger Count: {trigger_times}')
            
            if trigger_times >= patience:
                print('Early stopping activated.')
                return

if __name__ == '__main__':
    # 이미지와 각도 데이터를 불러옴
    dataset = ImageAngleDataset(image_dir, angle_data, transform=transform)

    # 학습 및 검증 데이터 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 모델 생성
    model = ResNetWithAngle()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)  # SGD with momentum and weight decay

    # 모델 학습
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=20)
