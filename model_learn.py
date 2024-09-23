import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# 얼굴 검출을 위한 OpenCV CascadeClassifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ResNet50 기반 모델 정의
class ResNet(nn.Module):
    def __init__(self, num_classes=3):
        super(ResNet, self).__init__()

        # torchvision에서 ResNet50의 사전 학습된 가중치를 불러옵니다.
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # 모든 레이어를 고정하여 학습되지 않도록 설정 (전이 학습)
        for param in self.model.parameters():
            param.requires_grad = False

        # 출력층 수정 (기본 출력층을 새로운 작업에 맞게 교체)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # 3개의 클래스로 출력
        )

    def forward(self, x):
        return self.model(x)

# 데이터셋 클래스 정의
class EyeRegionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # eye_region_0, eye_region_1, eye_region_2 폴더에서 이미지와 레이블을 로드
        for label in range(3):
            folder = f"eye_region_{label}"
            folder_path = os.path.join(root_dir, folder)
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg') or file_name.lower().endswith('.png'):
                    self.data.append((os.path.join(folder_path, file_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 얼굴 검출 및 크롭
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_image = image[y:y + h, x:x + w]
        else:
            # 얼굴 검출 실패 시 중앙 부분을 크롭
            h_img, w_img, _ = image.shape
            face_image = image[h_img//4: 3*h_img//4, w_img//4: 3*w_img//4]

        # PIL 이미지로 변환
        face_image = Image.fromarray(face_image)

        if self.transform:
            face_image = self.transform(face_image)

        return face_image, label

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50에 맞게 리사이즈
    transforms.ToTensor(),          # 텐서로 변환
    transforms.Normalize([0.485, 0.456, 0.406],  # 이미지 정규화
                         [0.229, 0.224, 0.225])
])

# 학습 및 검증 루프 설정
def train_model():
    # 데이터셋 준비
    dataset = EyeRegionDataset(root_dir='./', transform=transform)

    # 데이터셋을 9:1로 나누기 (Train 90%, Validation 10%)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 데이터로더 설정
    batch_size = 32  # VRAM 부담을 줄이기 위해 배치 크기 감소
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 장치 설정 (GPU 또는 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # 모델 초기화
    model = ResNet(num_classes=3).to(device)

    # 옵티마이저 설정
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    # 손실 함수
    criterion = nn.CrossEntropyLoss()

    # 혼합 정밀도 스케일러
    scaler = torch.amp.GradScaler()

    # Early Stopping을 위한 변수
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    # 학습 루프
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, labels in progress_bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            # autocast에 device_type 추가
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss / (progress_bar.n + 1))

        avg_train_loss = running_loss / len(train_loader)

        # Validation 루프
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

        # Early Stopping 체크
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # 성능이 개선되면 초기화
            # 성능이 개선될 때마다 모델 저장
            torch.save(model.state_dict(), 'resnet_tablet.pt')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print("학습 완료! 모델이 'resnet_tablet.pt'로 저장되었습니다.")

# 메인 블록에서 train_model 함수 호출
if __name__ == "__main__":
    train_model()
