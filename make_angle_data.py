import os
import cv2
import dlib
import numpy as np
import csv
from imutils import paths

# 폴더 경로 설정
input_dir = 'eye_region_data'
output_dir = 'eye_region_face'
csv_file = 'angle_data.csv'  # 각도 데이터를 저장할 CSV 파일 경로

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    for i in range(3):
        os.makedirs(f'{output_dir}/eye_region_face_{i}')

# dlib 모델 로드
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Head Pose 추정을 위한 3D 모델 좌표
model_points = np.array([
    (0.0, 0.0, 0.0),             # 코 끝 (nose tip)
    (0.0, -330.0, -65.0),         # 턱 끝 (chin)
    (-225.0, 170.0, -135.0),      # 왼쪽 눈 바깥 모서리 (left eye left corner)
    (225.0, 170.0, -135.0),       # 오른쪽 눈 바깥 모서리 (right eye right corner)
    (-150.0, -150.0, -125.0),     # 입 왼쪽 모서리 (left mouth corner)
    (150.0, -150.0, -125.0)       # 입 오른쪽 모서리 (right mouth corner)
], dtype=np.float64)

# 카메라의 내부 매개변수 (반으로 줄인 해상도에 맞춤)
size = (1016, 762)  # 줄인 해상도
focal_length = 762  # 줄인 높이를 기준으로 한 초점 거리
center = (508, 381)  # 줄인 이미지의 중심점

camera_matrix = np.array([
    [focal_length, 0, center[0]],
    [0, focal_length, center[1]],
    [0, 0, 1]
], dtype=np.float64)

# 왜곡 계수 (왜곡이 없다면 0으로 초기화)
dist_coeffs = np.zeros((4,1))

# CSV 파일 초기화 (헤더 작성)
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'yaw', 'pitch', 'roll'])  # 헤더 추가

# 얼굴과 시선 정보 전처리 함수 (증강 포함)
def preprocess_and_augment(image, face, output_folder, file_name):
    x, y, w, h = (face.left(), face.top(), face.width(), face.height())
    
    # 얼굴 영역 추출
    face_image = image[y:y+h, x:x+w]
    
    # 데이터 증강
    augmented_images = [face_image]
    
    # 밝기 조절: 밝기 높이고, 낮추는 두 가지 방식
    bright = cv2.convertScaleAbs(face_image, alpha=1.3, beta=30)  # 밝게
    dark = cv2.convertScaleAbs(face_image, alpha=0.7, beta=-30)  # 어둡게
    augmented_images.append(bright)
    augmented_images.append(dark)
    
    # 증강된 이미지 저장
    for i, aug_img in enumerate(augmented_images):
        output_path = os.path.join(output_folder, f"{file_name}_aug_{i}.jpg")
        cv2.imwrite(output_path, aug_img)

# 얼굴 각도 추출 함수
def get_head_pose(image, face, image_name):
    landmarks = predictor(image, face)
    landmarks_2D = np.array([
        (landmarks.part(30).x, landmarks.part(30).y),     # 코 끝
        (landmarks.part(8).x, landmarks.part(8).y),       # 턱 끝
        (landmarks.part(36).x, landmarks.part(36).y),     # 왼쪽 눈 바깥 모서리
        (landmarks.part(45).x, landmarks.part(45).y),     # 오른쪽 눈 바깥 모서리
        (landmarks.part(48).x, landmarks.part(48).y),     # 입 왼쪽 모서리
        (landmarks.part(54).x, landmarks.part(54).y)      # 입 오른쪽 모서리
    ], dtype=np.float64)
    
    # Head Pose 추정
    _, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, landmarks_2D, camera_matrix, dist_coeffs
    )
    
    # 회전 행렬 계산
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # 회전 행렬을 Yaw, Pitch, Roll로 변환
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
    yaw, pitch, roll = angles[1], angles[0], angles[2]
    
    # 각도 데이터를 CSV 파일에 저장
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_name, yaw, pitch, roll])
    
    return yaw, pitch, roll

# 데이터셋을 전처리 및 저장
for i in range(3):
    # eye_region_n 폴더의 이미지 파일 경로 리스트 가져오기
    image_paths = list(paths.list_images(os.path.join(input_dir, f'eye_region_{i}')))
    
    for image_path in image_paths:
        file_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 감지
        faces = detector(gray)
        
        if len(faces) > 0:
            for face in faces:
                # 얼굴 각도 추출 및 저장
                yaw, pitch, roll = get_head_pose(gray, face, file_name)
                
                # 전처리 및 증강 후 저장
                preprocess_and_augment(image, face, f'{output_dir}/eye_region_face_{i}', file_name)

print("데이터 전처리, 증강 및 각도 데이터 저장 완료!")
