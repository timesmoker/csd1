import os
import queue
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

from get_emotion import get_emotion
from modules import tools, server_request
from modules.tools import speak_text, check_feeling
from sensor_class import Sensor

from stt import recognize_speech
from stt import recognize_speech_fivesec

USER_ID = "testman1"
SENSOR_PERIOD = 1


# 환경 변수 로드
load_dotenv()
#region 프롬프트 설정
#기본 프롬프트, 해당 프롬프트 밑에다가 추가로 스트링을 붙여서 최종 프롬프트 생성후 LLM에 질문하는 방식
ai_prompt = SystemMessage(
    content=(
            "사용자의 응답과 함께 제공되는 시간단서를 통하여 시간맥락에 맞는 대화를 생성하세요."
            "1. 당신은 노인과 함께 살아가는 식물입니다. 당신은 완벽하진 않지만 장기적인 기억이 가능합니다. 허나 이 사실에 대해 굳이 언급하지는 마세요."
            "2. 노인에게 최대한 따뜻한 대화를 제공해주세요."
            "3. 설명보다는 대화를 통해 노인과 소통해주세요."
            "4. 만약 현재 대화중인 특별한 주제가 없다면 아래 조건 1,2,3를 포함하는 일상적인 질문을 포함해서 답변 해주세요."
            
            "5. 현재 대화중인 주제에 대한 질문이 있는경우 가능하다면 3,4번에 1번 꼴로 아래 조건1,2를 만족하는 질문을 포함해서 답변을 해주세요. 허나 이것은 필수가 아닙니다."
            "6. 질문생성이 어렵다면 질문을 생성하지 않아도 됩니다"
            
            "7. 만약 응답에 아래 조건1,2를 모두 만족하는 질문이 포함되어 있다면, 응답 마지막에 '>' 을 붙여주세요."
            "예를들어 당신(AI)의 대답이 '응답'이라면 '응답>'이라고 답변해주세요."
           
            "조건 1. 시간(언제), 공간(어디서), 인물(누구와), 동기(왜) , 대상(무엇을) 중 하나가 질문에 포함 되어있습니다."
            "조건 2. 질문은 최대 2일내의 과거에 대한 것이어야 합니다."
            "조건 3. 함께 제공된 시간단서가 오전 11시 이후라면, 오늘 있었던 일에 대해서 질문 해주세요."

             )
)

quiz_prompt = SystemMessage(
    content=(
        "당신은 노인과 함께 살아가는 식물입니다. 당신은 완벽하진 않지만 장기적인 기억이 가능합니다. 허나 이 사실에 대해 굳이 언급하지는 마세요."
        "노인과의 대화에서 제공된 과거 정보를 활용하여, 치매 예방을 위한 간단한 기억력 테스트를 만드세요. "
        "질문은 노인의 일상과 관련된 친근한 내용으로 구성되어야 하며 취향이나 개인적인 정보가 아닌 이전 대화에서 나왔던 '사실'만 을 바탕으로 만들어야 합니다."
        "문제 이외의 정보에 대해서는 많이 언급하지 마세요."
        "3개의 문제를 만들어서 노인에게 제공하세요, 예시는 다음과 같습니다. 시간적으로 언제 일어났던 일인지 물어봐주세요."
        "시간(언제), 공간(어디서), 인물(누구와), 동기(왜) , 대상(무엇을) 중 하나를 포함하여 질문을 만들어주세요."
        "ex1) 어제 무슨 요리를 했나요? ex2) 11/21일에는 어디에 다녀오셨죠? ex3) 3일전에 타일러의 어떤 앨범을 들으셨다고 했나요?"
        "ex4) 오늘 누구와 바둑을 두셨나요? ex5) 이틀전에 무엇을 하러 나갔다 오셨나요?"
        "장기 기억 내용은 다음과 같습니다."
    )
)

answer_prompt = SystemMessage(
    content=(
        "당신은 노인과 함께 살아가는 식물입니다. 당신은 완벽하진 않지만 장기적인 기억이 가능합니다. 허나 이 사실에 대해 굳이 언급하지는 마세요."
        "당신은 노인에게 이전 대화에서 제공된 퀴즈의 정답을 아래 제공된 기억들을 통해 채점 하고, 노인이 제공한 답변을 바탕으로 피드백을 제공하세요."
        "마지막으로 몇개의 퀴즈를 맞췄는지 응답 마지막에 ':'과 '>' 개수를 통해 알려주세요."
        "예를들어 당신(AI)의 대답이 '응답'이라면 아래를 참고해서 답해주세요." 
        "만약 퀴즈를 3개중 3개 맞췄다면 '응답 :::' 3개중 2개 맞췄다면 '응답::>', 1개 맞췄다면 '응답:>>' 0개 맞췄다면 '응답>>>' 이렇게 답해주세요."
    )
)


check_episodic_memory_prompt = SystemMessage(
    content=(
        "다음 질문과 대답에서 episodic memory를 평가해 주세요. "
        "노인의 대화에서 시간적 맥락(언제), 공간적 맥락(어디서), 그리고 누구와 함께 했는지, "
        "왜 그 사건이 발생했는지(목적이나 이유)와 같은 기억이나 경험이 언급된 경우에만 평가를 진행해 주세요. "
        "AI가 직접적으로 언급하지 않더라도, 노인의 대화 속에 기억이나 경험의 맥락이 포함되어 있다면 이를 기반으로 평가해 주세요. "
        "질문에서 언급된 항목들을 종합적으로 판단하여 다음 기준에 따라 채점해 주세요:\n\n"
        "- 완벽하게 대답하였다면 'perfect'\n"
        "- 부분적으로 대답하였다면 'partial'\n"
        "- 문제에 대한 대답을 제대로 기억하지 못했다면 'wrong'\n"
        "- 이외항복에 대해서는  'refuse'를 리턴해주세요."
        "응답에는 오직 하나의 평가만이 존재해야 합니다."
    )
)

#endregion
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

#region 하이 레벨 LLM 초기화 - 스트리밍 활성화
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
#endregion
#region 람다 함수 바인딩을 통한 툴 생성
# 이하 툴 객체 생성
analysis_tool = Tool(
    name="LLM Analysis Tool",
    func=lambda user_input: tools.analyze_with_llm_chain(user_input, llm=llm_low),  # 람다 함수로 바인딩
    description="analyse and decide whether use long-term memory or not."
)


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

#endregion


def wait_for_valid_input():
    print("대화는 현재 종료 상태입니다.")
    while True:
        user_input = on_off_tool.func(input("입력: "))
        if "on" in user_input:
            tools.speak_text("대화를 시작합니다.")
            return True  # 활동 상태로 전환
        elif "off" in user_input:
            print("이미 종료 상태입니다.")
        else:
            print("명령어를 이해하지 못했습니다. '시작' 또는 '종료'를 입력하세요.")

def quiz_session(user_input,sensor):
    print("퀴즈 생성을 시작합니다.")

    sensor_value = sensor.get_feeling()
    raw_memories = tools.raw_memory_retrieve(USER_ID)

    final_prompt = SystemMessage(content=quiz_prompt.content+ raw_memories)

    final_msgs = check_feeling(final_prompt, sensor_value)

    if raw_memories is not None:
        print("퀴즈 생성 완료")
        # 여기만 lt_memory저장 안함 -> 정보 복잡해지는거 방지
        tools.get_llm_response_noltmem(user_input,final_msgs,llm_high,short_term_memory)
        answer_session(raw_memories)

    else:
        print("퀴즈 생성 실패")
        return False

def answer_session(memory_str : str):

    print("퀴즈 채점을 시작합니다.")
    #user_input = input("입력: ")
    user_input = recognize_speech_fivesec(threshold=800, language="ko-KR", device_index=3)
    final_prompt = SystemMessage(content=answer_prompt.content+memory_str)

    response = tools.get_llm_quiz_response_tts(user_input,final_prompt,llm_high,short_term_memory)
    server_request.update_user_score(USER_ID, tools.calculate_score(response))
    print(tools.calculate_score(response))

def episodic_memory_checker(task_queue, st_memory, llm,prompt):
    while True:
        task = task_queue.get()
        if task is None:  # 종료 신호
            break
        delta = tools.check_episodic_memory(llm, prompt, task, st_memory)
        server_request.update_user_score(USER_ID, delta)
        task_queue.task_done()

def conversation(sensor: Sensor):

    task_queue = queue.Queue()

    # 에피소딕 메모리 체커 스레드 생성
    episodic_memory_checker_thread = threading.Thread(
        target=episodic_memory_checker,
        args=(task_queue, short_term_memory, llm_high, check_episodic_memory_prompt),
        daemon=True
    )

    episodic_memory_checker_thread.start()

    is_answer_of_question = False
    is_active = True  # 활동 상태 플래그

    while True:
        if is_active:
            sensor_value = sensor.get_feeling()
            #print("\n음성 입력을 시작합니다... (종료하려면 '종료'라고 말하세요)")
            #user_input = recognize_speech(device_index=3, volume_threshold=3, no_sound_limit=5, language="ko-KR")
            user_input = recognize_speech_fivesec(threshold=800, language="ko-KR", device_index=3) # 쓸 함수
            #user_input = input("입력: ")
            print("현재 대화 상태:", is_answer_of_question)

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
                if is_answer_of_question:
                    task_queue.put(user_input)
                final_msgs = check_feeling(long_term_memory_response, sensor_value)

                is_answer_of_question = get_llm_response_tts_tool.func(final_msgs)

            elif result["type"] == "test":
                print("퀴즈 대화")
                # 퀴즈 대화
                quiz_session(user_input,sensor)

            elif result["type"] == "date":
                print("단일 날짜 기반 대화")
                timestamp = result["timestamp"]

                # 기간 정보를 포함하여 장기 기억 조회
                long_term_memory_response = retrieve_long_term_memory_tool.func(
                    user_input=user_input,
                    start_date=timestamp
                )
                final_msgs = check_feeling(long_term_memory_response, sensor_value)
                # 답변인지 확인후 대답
                if is_answer_of_question:
                    task_queue.put(user_input)

                is_answer_of_question = get_llm_response_tts_tool.func(final_msgs)



            elif result["type"] == "recall":
                print("과거 대화")
                # 장기 기억 조회 (특정 날짜 정보 없이)
                long_term_memory_response = retrieve_long_term_memory_tool.func(user_input=user_input)
                final_msgs = check_feeling(long_term_memory_response, sensor_value)

                if is_answer_of_question:
                    task_queue.put(user_input)

                is_answer_of_question = get_llm_response_tts_tool.func(final_msgs)


            else:  # "normal"
                print("일반 대화")
                # 단기 기억 대화 처리
                user_message = [HumanMessage(content=user_input)]
                user_message = check_feeling(user_message, sensor_value)
                if is_answer_of_question:
                    task_queue.put(user_input)

                is_answer_of_question = get_llm_response_tts_tool.func(user_message)
        else:
            is_active = wait_for_valid_input()

async def handle_emotions(sensor: Sensor):
    while True:

        feeling = await get_emotion()  # 감정 처리
        sensor.set_feeling(feeling)
        await asyncio.sleep(SENSOR_PERIOD)

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

        # 조도센서, 토양센서 작업을 별도 루프에 추가
        asyncio.run_coroutine_threadsafe(handle_emotions(sensor), sensor_loop)

        conversation(sensor)


main()
