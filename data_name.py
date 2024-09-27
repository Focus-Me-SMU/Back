import os
from PIL import Image

# 원본 이미지 폴더 경로
folders = ['시선0', '시선1', '시선2']
# 새 폴더 이름
new_folders = ['eye_region_0_folder', 'eye_region_1_folder', 'eye_region_2_folder']
# 새 이미지 이름 접두사
prefixes = ['eye_region_0_', 'eye_region_1_', 'eye_region_2_']

# 각 폴더에 대해 처리
for folder, new_folder, prefix in zip(folders, new_folders, prefixes):
    os.makedirs(new_folder, exist_ok=True)  # 새 폴더 생성

    # 폴더 내의 모든 이미지 파일을 처리
    for i, filename in enumerate(os.listdir(folder)):
        if filename.endswith('.jpg'):
            # 이미지 열기
            img_path = os.path.join(folder, filename)
            with Image.open(img_path) as img:
                # 이미지 크기 변경
                img_resized = img.resize((1016, 762))
                
                # 새 이미지 이름 설정
                new_filename = f"{prefix}{i:03d}.jpg"
                new_img_path = os.path.join(new_folder, new_filename)
                
                # 이미지 저장
                img_resized.save(new_img_path)

print("이미지 이름과 크기 변경 완료!")