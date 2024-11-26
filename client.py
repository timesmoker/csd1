import os
import threading
import asyncio
from dotenv import load_dotenv

from mem0 import Memory

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_together import ChatTogether
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.tools import Tool

from modules import tools
from sensor_class import Sensor
from light import read_light_async

#from stt import recognize_speech

USER_ID = "testman1"
LIGHT_SENSOR_PERIOD = 1


# 환경 변수 로드
load_dotenv()

#기본 프롬프트, 해당 프롬프트 밑에다가 추가로 스트링을 붙여서 최종 프롬프트 생성후 LLM에 질문하는 방식
ai_prompt = SystemMessage(
    content=f"당신은 노인과 함께 살아가는 식물입니다. 당신은 완벽하진 않지만 장기적인 기억이 가능합니다."
            "노인에게 최대한 따뜻한 대화를 제공해주세요."
            "설명보다는 대화를 통해 노인과 소통해주세요."
)

# 롱텀메모리 설정
config = {
    # 임베디드 모델 설정 필요
    "llm": {
        "provider": "anthropic",
        "config": {
            "model": "claude-3-5-sonnet-latest",
            "temperature": 0.1,
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
long_term_memory = Memory.from_config(config)


# 단기 메모리 버퍼
short_term_memory = ConversationBufferMemory(
    memory_key="chat_history",
    buffer_size=16  # 대화 기록 개수 제한
)

#이하 LLM 설정

# 스트리밍 콜백 핸들러 설정
streaming_handler = StreamingStdOutCallbackHandler()


# 하이 레벨 LLM 초기화 - 스트리밍 활성화
llm_high = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    callbacks=[streaming_handler]  # 스트리밍 핸들러 추가
)

# 로우 레벨 LLM 초기화
llm_low = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=4,
    max_retries=2
)

# 이하 툴 객체 생성
analysis_tool = Tool(
    name="LLM Analysis Tool",
    func=lambda user_input: tools.analyze_with_llm_chain(user_input, llm=llm_low),  # 람다 함수로 바인딩
    description="analyse and decide whether use long-term memory or not."
)
# analyze_with_llm 함수에 llm을 바인딩하여 전달


short_term_chat_tool = Tool(
    name="Short Term Conversation",
    func=lambda user_input: tools.get_short_term_chat(user_input, llm=llm_high, memory=short_term_memory),
    description="short term memory based chatting."
)

retrieve_long_term_memory_tool = Tool(
    name="Long Memories retriever",
    func=lambda user_input, start_date=None, end_date=None: tools.get_long_term_memory(
        user_input,
        lt_memory=long_term_memory,
        user_id=USER_ID,
        start_date=start_date,
        end_date=end_date
    ),
    description="retrieve long-term memories with optional date filtering."
)

get_llm_response_tool = Tool(
    name="LLM Response Tool",
    func=lambda user_input: tools.get_llm_response(user_input, ai_prompt=ai_prompt, llm=llm_high, st_memory=short_term_memory, lt_memory=long_term_memory, user_id=USER_ID),
    description="get response from LLM."
)

get_llm_response_tts_tool = Tool(
    name="LLM Response Tool",
    func=lambda user_input: tools.get_llm_response_tts(user_input, ai_prompt=ai_prompt, llm=llm_high, st_memory=short_term_memory, lt_memory=long_term_memory, user_id=USER_ID),
    description="get response from LLM."
)

on_off_tool = Tool(
    name="On/Off tool",
    func=lambda user_input: tools.on_off_check_tool(user_input,llm=llm_low),
    description="turn on/off the chatbot."
)



def wait_for_valid_input():
    print("대화는 현재 종료 상태입니다. (다시 시작하려면 '시작'이라고 입력하세요)")
    while True:
        user_input = on_off_tool.func(input("입력: "))
        if "on" in user_input:
            tools.speak_text("대화를 시작합니다.")
            return True  # 활동 상태로 전환
        elif "off" in user_input:
            print("이미 종료 상태입니다.")
        else:
            print("명령어를 이해하지 못했습니다. '시작' 또는 '종료'를 입력하세요.")


def conversation(sensor: Sensor):

    is_active = True  # 활동 상태 플래그

    while True:
        if is_active:
           # print("\n음성 입력을 시작합니다... (종료하려면 '종료'라고 말하세요)")
           # user_input = recognize_speech(device_index=3, volume_threshold=3, no_sound_limit=5, language="ko-KR")

            user_input = input("입력: ")


            # 종료 조건 확인
            if "종료" in user_input:
                tools.speak_text("대화를 종료합니다.")
                is_active = False  # 활동 중단 상태로 전환
                continue

            print("입력된 질문:", user_input)    

            # 분석 툴 실행 (long-term memory 또는 short-term memory 선택)
            result = analysis_tool.func(user_input)  # analyze_with_llm_tool 함수 실행

            if result["type"] == "date_range":
                print("기간 기반 대화")
                start_timestamp = result["start_timestamp"]
                end_timestamp = result["end_timestamp"]

                # 기간 정보를 포함하여 장기 기억 조회
                long_term_memory_response = retrieve_long_term_memory_tool.func(
                    user_input=user_input,
                    start_date=start_timestamp,
                    end_date=end_timestamp
                )
                # LLM 응답 생성
                response = get_llm_response_tts_tool.func(long_term_memory_response)

            elif result["type"] == "date":
                print("단일 날짜 기반 대화")
                timestamp = result["timestamp"]

                # 단일 날짜 정보를 포함하여 장기 기억 조회
                long_term_memory_response = retrieve_long_term_memory_tool.func(
                    user_input=user_input,
                    start_date=timestamp
                )
                # LLM 응답 생성
                response = get_llm_response_tts_tool.func(long_term_memory_response)

            elif result["type"] == "recall":
                print("과거 대화")
                # 장기 기억 조회 (특정 날짜 정보 없이)
                long_term_memory_response = retrieve_long_term_memory_tool.func(user_input=user_input)
                # LLM 응답 생성
                response = get_llm_response_tts_tool.func(long_term_memory_response)

            else:  # "normal"
                print("일반 대화")
                # 단기 기억 대화 처리
                user_message = [HumanMessage(content=user_input)]
                response = get_llm_response_tts_tool.func(user_message)

            # 응답 출력
            if (response != None):
                print("응답:", response)
            else :
                print("응답이 없습니다.")

        else:
            # 종료 상태에서 대기 모드 진입
            is_active = wait_for_valid_input()

# 센서 작업을 주기적으로 실행하는 비동기 함수 이미 있는데 왜 또 만들었냐면, 여기다가 읽기 주기 넣는게 좋은 코드니까.
# 나중에 토양센서도 이렇게 추가하면 됨
async def handle_light(sensor: Sensor):
        while True:
            await read_light_async(sensor)
            await asyncio.sleep(LIGHT_SENSOR_PERIOD) # n초마다 조도값 읽기


def main():

        # 센서 객체 생성
        sensor = Sensor()

        # 센서 전용 이벤트 루프 생성
        sensor_loop = asyncio.new_event_loop()

        # 센서 이벤트 루프 실행 함수
        def run_sensor_loop():
            asyncio.set_event_loop(sensor_loop)
            print("센서 이벤트 루프 시작\n")
            sensor_loop.run_forever()

        # 별도의 스레드에서 조도 센서 이벤트 루프 실행
        sensor_thread = threading.Thread(target=run_sensor_loop, daemon=True)
        sensor_thread.start()

        # 조도 센서 작업을 별도 루프에 추가
        asyncio.run_coroutine_threadsafe(handle_light(sensor), sensor_loop)

        conversation(sensor)


main()