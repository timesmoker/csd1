from datetime import datetime
from mem0 import Memory
import asyncio


# 사실 여러개 붙여서 불러옴 걍 주면 알아서 시스템 프롬프트에 붙여서 쓸 것
def retrieve_context(memory: Memory, query: str, user_id: str, start_date: int = None, end_date: int = None):

    if end_date is not None:
        filters = {
            "created_at": {
                "gte": start_date,  # 시작 날짜
                "lte": end_date  # 끝 날짜
            }
        }
        print(filters)
        memories = memory.search(query, user_id=user_id, filters=filters)
    elif start_date is not None:
        end_date = start_date + 86400
        filters = {
            "created_at": {
                "gte": start_date,  # 시작 날짜
                "lte": end_date  # 끝 날짜
            }
        }
        print(filters)
        memories = memory.search(query, user_id=user_id, filters=filters)
    else:
        memories = memory.search(query, user_id=user_id)

    serialized_memories = ' '.join(
        f"{mem['memory']} (Date: {datetime.utcfromtimestamp(mem['created_at']).strftime('%Y-%m-%d')})"
        for mem in memories
    )

    print(serialized_memories)

    return serialized_memories


# 대화 형식으로 넣지만, 특정 사실 하나만 저장됨 , 중복되면 알아서 mem0가 알아서 처리해줌
def save_interaction(memory: Memory, input_user_id: str, user_input: str, assistant_response: str):
    user_id = input_user_id
    interaction = [
        {
            "role": "user",
            "content": user_input
        },
        {
            "role": "assistant",
            "content": assistant_response
        }
    ]
    memory.add(interaction,user_id)
