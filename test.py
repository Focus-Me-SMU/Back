import cv2
import requests

# 서버 주소 (백엔드 서버가 실행 중인 주소)
server_url = 'http://172.16.156.152:5000/upload-frame'

# 웹캠 열기
cap = cv2.VideoCapture(0)  # 0번 장치는 기본 웹캠

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

def send_frame_to_server(frame):
    # 프레임을 JPEG 형식으로 인코딩
    _, img_encoded = cv2.imencode('.jpg', frame)
    frame_bytes = img_encoded.tobytes()

    # 서버로 프레임 전송
    try:
        response = requests.post(server_url, files={'frame': frame_bytes})
        if response.status_code == 200:
            return response.json()
        else:
            print(f"서버 오류: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"서버와 통신 중 오류 발생: {e}")
    return None

try:
    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 가져올 수 없습니다.")
            break

        # 서버에 프레임 전송 및 결과 수신
        result = send_frame_to_server(frame)

        if result:
            print(f"서버로부터 받은 결과: {result}")

            # 페이지 완료 시
            if 'page_complete' in result:
                print(f"{result['page']}번 페이지를 모두 읽었어요. n을 입력해서 다음 페이지로 넘어가세요.")
                user_input = input("입력: ")
                if user_input == 'n':
                    print("다음 페이지로 이동합니다.")
                else:
                    print("올바른 입력을 해주세요.")

            # 모든 페이지 완료 시
            if 'full_complete' in result:
                print("모든 문장을 다 읽었습니다. f를 눌러서 프로그램을 종료하고 결과를 확인하세요.")
                user_input = input("입력: ")
                if user_input == 'f':
                    print("최종 결과를 요청 중입니다...")
                    break

        # 'q' 키를 누르면 웹캠 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 웹캠 해제
    cap.release()
    cv2.destroyAllWindows()

# 프로그램 종료 후 최종 통계 받기
final_report_url = server_url.replace("/upload-frame", "/final-report")
try:
    response = requests.post(final_report_url)
    if response.status_code == 200:
        final_report = response.json()

        # 데이터 추출
        program_start_time = final_report['program_start_time']
        program_end_time = final_report['program_end_time']
        program_duration = final_report['program_duration']
        average_cycle_time = final_report['average_cycle_time']
        total_frame_count = final_report['total_frame_count']
        non_awake_frame_count = final_report['non_awake_frame_count']

        # 집중도 계산
        concentration = ((total_frame_count - non_awake_frame_count) / total_frame_count) * 100

        # 보기 좋게 출력
        print(f"\n오늘 {program_start_time}에 읽기 시작해서 {program_end_time}에 읽는 것을 끝냈어요.\n")
        print(f"1문장을 읽을 때 평균 {average_cycle_time:.2f}초로 읽었고,\n")
        print(f"읽는 동안의 집중도는 {concentration:.2f}% 입니다.")

    else:
        print(f"서버로부터 최종 통계를 받을 수 없습니다: {response.status_code}")
except Exception as e:
    print(f"서버와의 통신 중 오류 발생: {e}")