import os
from mem0 import Memory
from dotenv import load_dotenv
import time

# 환경 변수 로드
load_dotenv()

# LLM 및 Vector Store 설정
config = {
    "llm": {
        "provider": "together",  # 'together'를 공급자로 설정
        "config": {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",  # 사용하려는 모델 지정
            "temperature": 0.2,
            "max_tokens": 1500,
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "CSD1",
            "api_key": os.environ["QDRANT_API_KEY"],
            "url": "https://b236ccb3-b4d5-4349-a48d-6a8026f62951.us-east4-0.gcp.cloud.qdrant.io",
            "port": 6333,
            "on_disk": True
        }
    }
}

# Memory 객체 초기화
m = Memory.from_config(config)

# 가짜 데이터 추가
fake_data = [
    {
        "role": "user",
        "content": "나는 한때 서울에 살았었지. 거기는 정말 아름다운 곳이었어"
    },
    {
        "role": "assistant",
        "content": "정말 아름다운 곳 입니다 서울, 저도 꼭 한번 보고싶네요! 서울의 어떤점이 좋으셨나요?."
                   ""
                   ""
    }
]

fake_data_2 = [
    {
        "role": "user",
        "content": "내가 한때 베트남에 있었어 전쟁이 있었지, 아주 끔찍한 경험이야 때론 꿈에 나오지 "
    },
    {
        "role": "assistant",
        "content": "전쟁은 참 어려운 경험이죠. 그런 경험을 하셨다니 정말 대단하십니다. 그런 경험을 통해 무엇을 배우셨나요?. 또 그런 경험을 통해 어떻게 변화되셨나요?. 저는 그런 경험을 통해 많은 것을 배우고 변화되셨다고 생각합니다."

    }
]

fake_data_3 = [
    {
        "role": "user",
        "content": "What's the best way to learn Python?"
    },
    {
        "role": "assistant",
        "content": "Start with online tutorials and practice daily."
    }
]


fake_data_4 = [
    {
        "role": "user",
        "content": "내 이름은 김덕수야"
    },
    {
        "role": "assistant",
        "content": "아, 김덕수님이시군요. 죄송합니다. 전에 제가 가진 정보에는 이름이 없었어요. 이제 알게 되어 기쁩니다. 김덕수님, 75세이시고 축구를 좋아하신다고 들었는데, 어떤 축구팀을 응원하시나요? 아니면 직접 축구를 즐겨 하시나요?."
    }
]

fake_data_5 = [
    {
        "role": "user",
        "content": "My name is \"Kim Deoksoo\""
    },
    {
        "role": "assistant",
        "content": "Ah, you are Mr. Kim Deoksoo. I apologize, I didn’t have your name in my previous information. I’m glad to know now. I heard you are 75 years old and like soccer. Which soccer team do you support? Or do you enjoy playing soccer yourself?"
    }
]

# 데이터 추가 함수
def add_fake_data(memory, user_id, interactions):
    memory.add(interactions, user_id=user_id)

# 사용자 ID
user_id = "manman"

# 가짜 데이터 추가
add_fake_data(m, user_id, fake_data_2)

# 타이밍 시작
start_time = time.time()

# 메모리 검색
query = "내가 베트남 얘기 했나? ."
related_memories = m.search(query=query, user_id=user_id)


# 타이밍 종료
end_time = time.time()

# 검색 결과 출력
print("Related Memories:")
print(related_memories)

# 소요된 시간 출력
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.4f} seconds")
