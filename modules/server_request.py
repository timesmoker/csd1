import requests
import json

# 서버 URL
base_url = "http://52.78.34.60:5986"  # Flask 서버 주소 입력 (IP 또는 도메인)


def get_user_data(username):
    url = f"{base_url}/get_data/{username}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print("사용자 데이터 가져오기 성공:")
            print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        else:
            print("GET 요청 실패:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("GET 요청 중 오류 발생:", e)

def update_user_data(username, updated_data):
    url = f"{base_url}/update_data/{username}"
    try:
        response = requests.put(url, json=updated_data)
        if response.status_code == 200:
            print("사용자 데이터 업데이트 성공:")
            print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        else:
            print("PUT 요청 실패:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("PUT 요청 중 오류 발생:", e)

def add_user(username, new_user_data):
    url = f"{base_url}/add_user/{username}"
    try:
        response = requests.post(url, json=new_user_data)
        if response.status_code == 200:
            print("새로운 사용자 추가 성공:")
            print(json.dumps(response.json(), indent=4, ensure_ascii=False))
        else:
            print("POST 요청 실패:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("POST 요청 중 오류 발생:", e)

username = "alice"
get_user_data(username)