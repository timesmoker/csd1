import os

from mem0 import Memory
from dotenv import load_dotenv
import time
# 환경 변수 로드
load_dotenv()

# LLM 설정
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
            "api_key":os.environ["QDRANT_API_KEY"],
            "url": "https://b236ccb3-b4d5-4349-a48d-6a8026f62951.us-east4-0.gcp.cloud.qdrant.io",
            "port": 6333,
            "on_disk": True
        }
    }
}

# Memory 객체 초기화
m = Memory.from_config(config)


start_time = time.time()
# 메모리 검색
related_memories = m.search(query="Help me plan my weekend.", user_id="alice")

# 타이밍 종료
end_time = time.time()

# 검색 결과 출력
print(related_memories)

# 소요된 시간 출력
elapsed_time = end_time - start_time
print(f"Time taken: {elapsed_time:.4f} seconds")
