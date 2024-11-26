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
from modules.tools import speak_text
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

quiz_prompt = SystemMessage(
    content=(
        "당신은 노인과 함께 살아가는 식물입니다. 당신은 완벽하진 않지만 장기적인 기억이 가능합니다. "
        "노인과의 대화에서 제공된 과거 정보를 활용하여, 치매 예방을 위한 간단한 기억력 테스트를 만드세요. "
        "질문은 노인의 일상과 관련된 친근한 내용으로 구성되어야 하며 취향이나 개인적인 정보가 아닌 이전 대화에서 나왔던 '사실'만 을 바탕으로 만들어야 합니다."
        "3개의 문제를 만들어서 노인에게 제공하세요, 예시는 다음과 같습니다. 시간적으로 언제 일어났던 일인지 물어봐주세요."
        "ex1) 어제 무슨 요리를 했나요? ex2) 11/21일에는 어디에 다녀오셨죠? ex3) 2주전에 저와 음악에 대해 이야기하신게 기억 나시나요?"
        "ex4) 3주전에 타일러의 어떤 앨범을 들으셨다고 했나요? ex5) 어제 져녁에 게장을 먹은게 기억 나시나요?"
        "장기 기억 내용은 다음과 같습니다."
    )
)

answer_prompt = SystemMessage(
    content=(
        "당신은 노인과 함께 살아가는 식물입니다. 당신은 완벽하진 않지만 장기적인 기억이 가능합니다. "
        "당신은 노인에게 이전 대화에서 제공된 퀴즈의 정답을 아래 제공된 기억들을 통해 채점 하고, 노인이 제공한 답변을 바탕으로 피드백을 제공하세요."
        "마지막으로 몇개의 퀴즈를 맞췄는지 응답 마지막에 ':'과 ';' 개수를 통해 알려주세요."
        "만약 퀴즈를 3개중 3개 맞췄다면 '응답:::' 3개중 2개 맞췄다면 '응답::;', 1개 맞췄다면 '응답:;;' 0개 맞췄다면 '응답;;;' 이렇게 답해주세요."
    )
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

def quiz_session():
    print("퀴즈 시작")


def conversation(sensor: Sensor):

    is_active = True  # 활동 상태 플래그

    while True:
        if is_active:
           # print("\n음성 입력을 시작합니다... (종료하려면 '종료'라고 말하세요)")
           # user_input = recognize_speech(device_index=3, volume_threshold=3, no_sound_limit=5, language="ko-KR")

            user_input = input("입력: ")

            print("입력된 질문:", user_input)    

            # 분석 툴 실행 (long-term memory 또는 short-term memory 선택)
            result = analysis_tool.func(user_input)  # analyze_with_llm_tool 함수 실행

            if result["type"] == "quit":
                print("대화를 중지합니다.")
                speak_text("대화를 종료합니다")
                is_active = False

            elif result["type"] == "date_range":
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

            elif result["type"] == "quiz":
                print("퀴즈 대화")
                # 퀴즈 대화
                response = get_llm_response_tts_tool.func(user_input)

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