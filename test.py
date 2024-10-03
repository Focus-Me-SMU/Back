import cv2
import requests
import numpy as np

# Flask 서버의 주소
url = 'http://127.0.0.1:5000/upload-frame'

# 웹캠 초기화
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

next_page = False  # 'next' 신호가 들어왔을 때 플래그 설정
diary_completed = False  # 일기가 완료되었는지 여부를 위한 플래그 설정

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break

    # 프레임을 JPEG로 인코딩
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    # 인코딩된 이미지를 Flask 서버로 전송
    try:
        response = requests.post(url, files={'frame': img_encoded.tobytes()})
        data = response.json()

        # 서버로부터 받은 응답 확인
        print(f"Server response: {data}")

        # 'next' 신호를 받은 경우
        if data.get('message') == 'next':
            next_page = True
            print("Next page signal received. Press 'n' to go to the next page.")

        # 모든 작업이 완료되었을 경우 처리
        if diary_completed:
            print("Diary completed. Exiting...")
            break

    except Exception as e:
        print(f"Error: {e}")
    
    # 프레임을 보여줌
    cv2.imshow('Webcam', frame)

    # 'n' 키를 누르면 다음 페이지로 넘어감
    if next_page and cv2.waitKey(1) & 0xFF == ord('n'):
        print("Moving to the next page...")
        next_page = False  # 플래그 초기화

        # 마지막 페이지일 경우 종료
        if diary_completed:
            print("Exiting...")
            break

    # 종료 조건: 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 릴리즈
cap.release()
cv2.destroyAllWindows()
