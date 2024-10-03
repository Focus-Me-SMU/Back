import pymysql

# 1. mysql 연결
conn = pymysql.connect(host='127.0.0.1', user='root', password='root', db='FocusMe', charset='utf8', port=3305)

# 2. 커서 생성하기
cur = conn.cursor()


# # 5. 입력한 데이터 저장하기
# conn.commit()

# # 6. MySQL 연결 종료하기
# conn.close()