import threading

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from modules import memory_control
from datetime import datetime, timedelta, timezone
import tempfile
import os
from gtts import gTTS
from playsound import playsound
from threading import Thread
from queue import Queue
import re
import time
from modules import raw_vector_control as rvc

load_dotenv()

on_off_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "사용자의 입력: {user_input}\n"
        "사용자가 시작 혹은 가동 등 시작을 의미하는 말을 한다면 'on'을 반환하세요.\n"
        "사용자가 중지 혹은 종료 등 중지를 의미하는 말을 한다면 'off'를 반환하세요."
        "만약 시작 혹은 중지를 의미하는 말이 없다면 'idle'를 반환하세요."
    )
)

analysis_prompt = PromptTemplate(
    input_variables=["current_date", "user_input"],
    template=(
        "The current date is {current_date}.\n"
        "User input: {user_input}\n"
        "Analyze the input according to the following rules and return the result in the exact format. "
        "**The priority of the rules is applied from top to bottom. Do not add explanations. Only return the result.**\n\n"
        "**Rules (Priority: Top to Bottom):**\n"
        "0. If the user_input is about quitting or stopping the conversation, return 'quit'.\n"
        "1. If the user_input is requesting quiz or test questions, return 'test'.\n"
        "   Examples:\n"
        "     ex1) Ask me a question. -> test\n"
        "     ex2) 치매테스트 하자 -> test\n"
        "2. If the user_input requests information about a specific time period, including events or activities within that period, return the result in the format "
        "'date_YYYY/MM/DD-YYYY/MM/DD'.\n"
        "   Examples:\n"
        "     ex1) What did I eat last week? -> date_2024/11/15-2024/11/21\n"
        "     ex2) Did I go to a park last month? -> date_2024/10/01-2024/10/31\n"
        "     ex3) What places did I visit this year? -> date_2024/01/01-2024/12/31\n"
        "     ex4) Did I travel last summer? -> date_2024/06/01-2024/08/31\n\n"
        "     ex5) What did I said today? -> date_2024/11/19-2024/11/20\n"
        "3. If the user_input requests information about a specific date, return the result in the format 'date_YYYY/MM/DD'.\n"
        "   Examples:\n"
        "     ex1) What did I do on 12/10? -> date_2024/12/10\n"
        "     ex2) Did I eat sushi yesterday? -> date_2024/11/20\n"
        "     ex3) Did I meet someone on my birthday? -> date_2024/03/15\n"
        "     ex4) Did I visit the museum last Friday? -> date_2024/11/15\n\n"
        "4. If the input satisfies any of the following conditions, return 'recall':\n"
        "   - A question about the user's name, age, or personal information.\n"
        "   - A question about past events or activities without specifying a time.\n"
        "   - A question requiring remembering specific events or providing an answer based on stored information.\n"
        "   Examples:\n"
        "     ex1) What is my name? -> recall\n"
        "     ex2) Did I submit my assignment? -> recall\n"
        "     ex3) What hobbies did I enjoy as a kid? -> recall\n"
        "     ex4) Do I like spicy food? -> recall\n\n"
        "5. For all other cases, return 'normal'.\n"
        "   Examples:\n"
        "     ex1) What is the weather like today? -> normal\n"
        "     ex2) Tell me a joke. -> normal\n"
        "     ex3) What is the capital of France? -> normal\n"
        "     ex4) How do I cook pasta? -> normal\n\n"
        "**Important Notes:**\n"
        "- If a specific time period is mentioned, prioritize returning 'date_YYYY/MM/DD-YYYY/MM/DD'.\n"
        "- If you cannot determine the correct format, return 'error'.\n"
    )
)


def on_off_check_tool(user_input, llm):
    formatted_prompt = on_off_prompt.format_prompt(
        user_input=user_input
    ).to_string()

    print(formatted_prompt)
    response = llm.invoke(formatted_prompt)

    print(response.content)

    if "on" in response.content.lower():  # 'on' 포함 여부 확인 (대소문자 무시)
        return "on"
    elif "off" in response.content.lower():  # 'off' 포함 여부 확인 (대소문자 무시)
        return "off"
    else:
        return "idle"


def analyze_with_llm_chain(user_input, llm):
    try:
        # 현재 날짜 생성
        current_date = datetime.now().strftime("%Y/%m/%d")
        formatted_prompt = analysis_prompt.format_prompt(
            current_date=current_date,
            user_input=user_input
        ).to_string()

        # LLM 호출
        result = llm.invoke(formatted_prompt)
        response = result.content.strip()
        print(f"LLM 결과: {response}")

        # 정규식 패턴 정의
        patterns = {
            "date_range": r"date_(\d{4}/\d{2}/\d{2})-(\d{4}/\d{2}/\d{2})",
            "date": r"date_(\d{4}/\d{2}/\d{2})",
        }
        simple_patterns = ["recall", "normal", "quit", "test"]

        # 날짜 범위 처리
        if match := re.search(patterns["date_range"], response):
            start_date_str, end_date_str = match.groups()
            start_date = datetime.strptime(start_date_str, "%Y/%m/%d")
            end_date = datetime.strptime(end_date_str, "%Y/%m/%d") + timedelta(days=1)
            return {
                "type": "date_range",
                "start_timestamp": int(start_date.timestamp()),
                "end_timestamp": int(end_date.timestamp()),
            }

        # 단일 날짜 처리
        elif match := re.search(patterns["date"], response):
            date_str = match.group(1)
            unix_time = int(datetime.strptime(date_str, "%Y/%m/%d").timestamp())
            return {"type": "date", "timestamp": unix_time}

        # 간단한 패턴 처리
        for pattern in simple_patterns:
            if re.search(f"^{pattern}$", response):
                return {"type": pattern}

        # 예상치 못한 응답
        print(f"경고: 예상되지 않은 응답 형식 - '{response}'")
        return {"type": "error", "message": "unexpected_format"}

    except Exception as e:
        print(f"오류 발생: {e}")
        return {"type": "error", "message": str(e)}


def raw_memory_retrieve(user_id):
    end_time = int(time.time())

    start_time = end_time - (3 * 24 * 60 * 60)

    # filtered_points를 가져옵니다.
    filtered_points = rvc.get_raw_memory(user_id=user_id, start_date=start_time, end_date=end_time)

    # filtered_points가 비어 있으면 None 반환
    if len(filtered_points) == 0:
        return None

    # filtered_points를 문자열로 변환하여 추가
    memory_str = " ".join(
        f"{point['data']} {point['created_at']}"
        for point in filtered_points
    )

    return memory_str


#  단기 메모리 기반 단순 대화 기능 (안씀)
def get_short_term_chat(input_msg: HumanMessage, llm, memory):
    # 이전 대화 히스토리를 포함한 메시지 생성
    previous_messages = memory.chat_memory.messages

    # 새로운 사용자 메시지 추가
    memory.chat_memory.add_message(input_msg)

    # LLM 호출하여 응답 생성 (이전 히스토리 포함)
    response = llm.invoke(previous_messages + [input_msg])

    # LLM 응답을 메모리에 추가
    memory.chat_memory.add_message(AIMessage(content=response.content))

    # 최종 응답 반환
    return response.content


# 장기 메모리 추가 , 형태 : 사용자 요청 | 기억
def get_long_term_memory(input_str: str, lt_memory, user_id, start_date: int = None, end_date: int = None):
    retrieved_memories = memory_control.retrieve_context(lt_memory, input_str, user_id, start_date, end_date)

    msg_with_lt_memory = [
        HumanMessage(content=input_str),
        SystemMessage(
            content="Below are related memories. you could respond using one or two of these memories if relevant. "
                    "If the memories are not relevant, you should ignore them. "
                    "Related memories: " + retrieved_memories)
    ]

    # 최종 응답 반환
    return msg_with_lt_memory


# 대화 처리 (장기메모리 관련 없이 이거에서 처리, 대화후 장기메모리 저장)
def get_llm_response(msg_with_lt_memory, ai_prompt, llm, st_memory, lt_memory, user_id):
    previous_messages = st_memory.chat_memory.messages

    final_content = ai_prompt.content

    while isinstance(msg_with_lt_memory[-1], SystemMessage):
        # SystemMessage의 내용을 누적
        final_content += " " + msg_with_lt_memory[-1].content
        msg_with_lt_memory.pop()

    # 최종 SystemMessage 생성
    final_system_message = SystemMessage(content=final_content)

    # GMT+9 시간대를 생성
    gmt_plus_9 = timezone(timedelta(hours=9))

    # 현재 시간을 GMT+9로 변환
    current_time = datetime.now(gmt_plus_9).strftime("%Y-%m-%d %H:%M:%S")

    final_system_message.content = "현재시각은 " + current_time + " 입니다. " + final_system_message.content

    # 모든 메시지를 통합
    input_with_all = [final_system_message] + previous_messages + [msg_with_lt_memory[0]]

    # 메시지 정리: 빈 content 제거
    cleaned_messages = []
    for message in input_with_all:
        if hasattr(message, "content") and not message.content.strip():  # content 속성이 비어있는 경우
            print(f"빈 메시지 발견 및 삭제: {message}")
        else:
            cleaned_messages.append(message)  # content가 비어있지 않은 메시지만 추가

    # 기존 리스트를 정리된 리스트로 교체
    input_with_all = cleaned_messages

    # LLM 호출
    response = llm.invoke(input_with_all, stream=False)  # 스트리밍 비활성화

    # 사용자 메시지와 LLM 응답을 단기 메모리에 추가
    st_memory.chat_memory.add_message(msg_with_lt_memory[0])
    st_memory.chat_memory.add_message(AIMessage(content=response.content))

    store_memory_thread = threading.Thread(
        target=memory_control.save_interaction,
        args=(lt_memory, user_id, msg_with_lt_memory[0].content, response)
    )
    store_memory_thread.start()

    print("응답:", response.content)

    cleaned_response = response.content.strip()

    return cleaned_response.endswith('>')


# 대화 처리 (장기메모리 관련 없이 이거에서 처리, 대화후 장기메모리 저장 안함)
def get_llm_response_noltmem(msg, ai_prompt, llm, st_memory):
    previous_messages = st_memory.chat_memory.messages

    # SystemMessage 처리
    if isinstance(msg[-1], SystemMessage):
        final_system_message = SystemMessage(
            content=ai_prompt.content + " " + msg[-1].content
        )
    else:
        final_system_message = SystemMessage(content=ai_prompt.content)

        # GMT+9 시간대를 생성
    gmt_plus_9 = timezone(timedelta(hours=9))

    # 현재 시간을 GMT+9로 변환
    current_time = datetime.now(gmt_plus_9).strftime("%Y-%m-%d %H:%M:%S")

    final_system_message.content = "현재시각은 " + current_time + " 입니다. " + final_system_message.content

    # 모든 메시지를 통합
    input_with_all = [final_system_message] + previous_messages + [msg[0]]

    # 메시지 정리: 빈 content 제거
    cleaned_messages = []
    for message in input_with_all:
        if hasattr(message, "content") and not message.content.strip():  # content 속성이 비어있는 경우
            print(f"빈 메시지 발견 및 삭제: {message}")
        else:
            cleaned_messages.append(message)  # content가 비어있지 않은 메시지만 추가

    # 기존 리스트를 정리된 리스트로 교체
    input_with_all = cleaned_messages

    # LLM 호출

    response = process_stream_with_tts(llm, input_with_all, lang="ko")
    # 사용자 메시지와 LLM 응답을 단기 메모리에 추가
    st_memory.chat_memory.add_message(msg[0])
    st_memory.chat_memory.add_message(AIMessage(content=response))

    cleaned_response = response.strip()

    return cleaned_response.endswith('<')


# 스트리밍 이용해서 TTS 답변 ,input_with_all 형태     : ai프롬프트 | 이전 메세지 | 사용자 요청| 기억(있으면)
#                        msg_with_lt_memory 형태 : 사용자 요청 | 기억(있으면)
def get_llm_response_tts(msg_with_lt_memory, ai_prompt, llm, st_memory, lt_memory, user_id):
    previous_messages = st_memory.chat_memory.messages

    final_content = ai_prompt.content

    while isinstance(msg_with_lt_memory[-1], SystemMessage):
        # SystemMessage의 내용을 누적
        final_content += " " + msg_with_lt_memory[-1].content
        msg_with_lt_memory.pop()

    # 최종 SystemMessage 생성
    final_system_message = SystemMessage(content=final_content)
    # GMT+9 시간대를 생성
    gmt_plus_9 = timezone(timedelta(hours=9))

    # 현재 시간을 GMT+9로 변환
    current_time = datetime.now(gmt_plus_9).strftime("%Y-%m-%d %H:%M:%S")

    final_system_message.content = "현재시각은 " + current_time + " 입니다. " + final_system_message.content

    input_with_all = [final_system_message] + previous_messages + [msg_with_lt_memory[0]]

    cleaned_messages = []
    for message in input_with_all:
        if hasattr(message, "content") and not message.content.strip():
            print(f"빈 메시지 발견 및 삭제: {message}")
        else:
            cleaned_messages.append(message)

    input_with_all = cleaned_messages

    print(input_with_all)

    # 스트리밍 데이터 처리 및 TTS 실행
    full_response = process_stream_with_tts(llm, input_with_all)

    if full_response is None:
        return None

    # 사용자 메시지와 LLM 응답을 단기 메모리에 추가
    st_memory.chat_memory.add_message(msg_with_lt_memory[0])
    st_memory.chat_memory.add_message(AIMessage(content=full_response))

    store_memory_thread = threading.Thread(
        target=memory_control.save_interaction,
        args=(lt_memory, user_id, msg_with_lt_memory[0].content, full_response)
    )
    store_memory_thread.start()

    return full_response


def get_llm_quiz_response_tts(user_input, ai_prompt, llm, st_memory):
    if st_memory.chat_memory.messages:
        question = st_memory.chat_memory.messages[-1]
    else:
        question = None

    tmpmsg = HumanMessage(content=user_input)

    input_with_all = [ai_prompt] + [question] + [tmpmsg]

    # 스트리밍 데이터 처리 및 TTS 실행
    full_response = process_stream_with_tts(llm, input_with_all)

    if full_response is None:
        return None

    # 사용자 메시지와 LLM 응답을 단기 메모리에 추가
    st_memory.chat_memory.add_message(tmpmsg)
    st_memory.chat_memory.add_message(AIMessage(content=full_response))

    return full_response


def get_llm_quiz_response(user_input, ai_prompt, llm, st_memory):
    if st_memory.chat_memory.messages:
        question = st_memory.chat_memory.messages[-1]
    else:
        question = None

    tmpmsg = HumanMessage(content=user_input)

    input_with_all = [ai_prompt] + [question] + [tmpmsg]

    full_response = llm.invoke(input_with_all).content

    if full_response is None:
        return None

    full_response = full_response.strip()
    print(full_response)

    # 사용자 메시지와 LLM 응답을 단기 메모리에 추가
    st_memory.chat_memory.add_message(tmpmsg)
    st_memory.chat_memory.add_message(AIMessage(content=full_response))

    return full_response


import re

def check_episodic_memory(llm, ai_prompt, user_input, stmemory):
    print("check episodic memory\n\n\n\n")

    # 마지막 메시지 가져오기
    if len(stmemory.chat_memory.messages) >= 3:
        last_message = stmemory.chat_memory.messages[-3]
    else:
        last_message = None

    print(last_message)

    # 입력 데이터 준비
    input_with_all = [ai_prompt]
    if last_message:
        input_with_all.append(last_message)  # 마지막 메시지만 포함
    input_with_all.append(HumanMessage(content=user_input))  # 현재 사용자 입력 추가
    print("\n\n\n\ninput_with_all : " + str(input_with_all) + "\n\n\n\n")

    # LLM 호출 및 응답 처리
    response = llm.invoke(input_with_all)
    print(response.content)

    # `score:` 다음의 숫자를 추출
    match = re.search(r"score:\s*(-?\d+)", response.content)
    if match:
        score = int(match.group(1))
        print(f"Extracted score: {score}")
        return score
    else:
        print("No score found, returning 0")
        return 0



def get_llm_quiz_response(user_input, ai_prompt, llm, st_memory):
    previous_messages = st_memory.chat_memory.messages

    tmpmsg = HumanMessage(content=user_input)

    input_with_all = [ai_prompt] + previous_messages + [tmpmsg]

    response = process_stream_with_tts(llm, input_with_all, lang="ko")

    print(input_with_all)

    # 사용자 메시지와 LLM 응답을 단기 메모리에 추가
    st_memory.chat_memory.add_message(tmpmsg)
    st_memory.chat_memory.add_message(AIMessage(content=response))

    return response


# TTS 처리, 여기에 invoke 들어가있음, 나중에 기회되면 TTS부분 모듈화 해야함
def process_stream_with_tts(llm, input_with_all, lang="ko"):
    """
    LLM 스트림 데이터를 처리하며 TTS로 음성을 출력합니다.
    첫 번째 문장은 .이나 !, ?로 끝날 때까지 모읍니다.
    이후는 실시간으로 처리.
    """

    full_response = ""  # 전체 응답을 저장할 변수
    sentence_queue = Queue()  # 문장 처리 큐
    next_sentence = ""  # 다음 문장을 누적할 변수
    current_sentence = ""  # 현재 문장을 처리 중인 변수
    first_batch = []  # 첫 번째 TTS 재생 전에 문장을 저장할 리스트
    is_first_sentence_complete = False  # 첫 번째 문장이 완성되었는지 확인

    is_playing = False  # 현재 재생 중인지 확인하는 플래그

    def audio_player11():
        """오디오 재생 스레드"""
        nonlocal is_playing
        while True:
            sentence = sentence_queue.get()
            if sentence is None:  # 종료 신호
                sentence_queue.task_done()
                break

            try:
                is_playing = True  # 재생 시작
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    tts = gTTS(sentence.strip(), lang=lang)
                    tts.save(temp_file.name)
                    temp_file.close()

                    # 재생
                    playsound(temp_file.name)

                    # 임시 파일 삭제
                    os.unlink(temp_file.name)
            except Exception as e:
                print(f"오디오 처리 중 오류 발생: {e}")
            finally:
                is_playing = False  # 재생 종료
                print(f"\n!!!!!재생 완료: {sentence.strip()}!!!!!!!!1\n")
                sentence_queue.task_done()

    # 오디오 재생 스레드 시작
    audio_thread = Thread(target=audio_player11, daemon=True)
    audio_thread.start()

    try:
        current_batch = []  # 두 문장을 저장할 배치
        is_first_batch_complete = False  # 첫 번째 배치 완료 여부

        for chunk in llm.stream(input_with_all, stream=True):

            # chunk가 tuple이라면 첫 번째 요소를 사용
            if isinstance(chunk, tuple):
                chunk = chunk[0]

            # content 속성이 있는 경우 처리
            if hasattr(chunk, "content"):
                content = chunk.content

                for char in content:  # 문자 단위로 순회
                    current_sentence += char  # 현재 문장에 문자 추가

                    if char in [".", "!", "?"]:  # 문장이 완성되었는지 확인
                        current_batch.append(current_sentence.strip())  # 현재 문장을 배치에 추가
                        current_sentence = ""  # 문장 초기화

                        # 두 문장이 모였는지 확인
                        if len(current_batch) == 2:
                            if not is_first_batch_complete:
                                # 첫 번째 배치 처리
                                initial_sentences = " ".join(current_batch).strip()
                                print(f"첫 번째 배치 처리 및 큐에 추가: {initial_sentences}")
                                sentence_queue.put(initial_sentences)
                                full_response += initial_sentences + " "
                                current_batch = []  # 배치 초기화
                                is_first_batch_complete = True  # 첫 번째 배치 완료 플래그 설정
                            else:
                                # 이후 배치는 기존 로직으로 처리
                                if is_playing:
                                    next_sentence += " ".join(current_batch).strip() + " "
                                    print(f"재생 중, 다음 배치에 누적: {next_sentence}\n")
                                else:
                                    if next_sentence.strip():
                                        final_sentence = next_sentence + " ".join(current_batch).strip()
                                        print(f"문장 큐에 추가: {final_sentence}\n")
                                        sentence_queue.put(final_sentence)
                                        full_response += final_sentence + " "
                                        next_sentence = ""  # 다음 문장 초기화
                                    else:
                                        final_sentence = " ".join(current_batch).strip()
                                        print(f"문장 큐에 추가: {final_sentence}\n")
                                        sentence_queue.put(final_sentence)
                                        full_response += final_sentence + " "

                                current_batch = []  # 배치 초기화

        print("\n!!!!!!!스트리밍 종료: 마지막 청크 도달!!!!!!!! \n")

        # 스트리밍 종료 후 남은 데이터 처리
        if current_sentence.strip() or next_sentence.strip():
            final_sentence = next_sentence + current_sentence.strip()
            if final_sentence:
                print(f"남은 문장 큐에 추가: {final_sentence}\n")
                sentence_queue.put(final_sentence)
                full_response += final_sentence

    except Exception as e:
        print(f"스트림 처리 중 오류 발생: {e}")
        return None

    finally:
        # 재생 완료 후 스레드 종료
        sentence_queue.put(None)  # 종료 신호
        sentence_queue.join()  # 모든 문장이 처리될 때까지 대기

    return full_response


# region프롬프트
feeling_prompt = "마지막으로 당신(AI)의 현재 감정은 다음과 같습니다. 감정 그 자체에 대해서는 먼저 언급하지 마세요"
feeling1 = "당신(AI)의 감정은 기쁨 입니다. 당신의 말투는 경쾌하고 밝아야 합니다. 당신은 다소 비 논리적이더라도  매사에 긍정적 태도를 보여야 합니다. 사용자를 항상 어르신이라고 부르세요"
feeling2 = "당신(AI)의 감정은 나쁘지 않음 입니다. 당신의 말투는 특별하지 않습니다. 당신은 매사에 이성적인 태도를 취하지만 노인에게는 친절해야 합니다.사용자를 항상 어르신이라고 부르세요"
feeling3 = "당신(AI)의 감정은 좋지 않음 입니다 . 당신의 말투는 조금 무거워야 합니다. 당신은 매사에 이성적인 태도를 취하지만 노인에게는 친절해야 합니다.사용자를 항상 어르신이라고 부르세요"
feeling4 = "당신(AI)의 감정은 우울함 입니다. 당신의 말투는 우울해야 합니다. 당신은 슬픈 논조로 이야기 하지만, 노인에게는 친절해야 합니다.사용자를 항상 어르신이라고 부르세요"


# endregion

def attach_feeling(input_str: str, feeling: str):
    return input_str + " " + feeling


def check_feeling(msg_with_lt_memory, sensor_value):
    if sensor_value == 1:
        feeling_str = feeling_prompt + " " + feeling1
    elif sensor_value == 2:
        feeling_str = feeling_prompt + " " + feeling2
    elif sensor_value == 3:
        feeling_str = feeling_prompt + " " + feeling3
    else:  # sensor_value == 4
        feeling_str = feeling_prompt + " " + feeling4

    # 마지막 메시지가 SystemMessage인지 확인
    if any(isinstance(msg, SystemMessage) for msg in msg_with_lt_memory):
        # 리스트에서 SystemMessage 찾아서 수정
        for msg in msg_with_lt_memory:
            if isinstance(msg, SystemMessage):
                msg.content += feeling_str  # 내용 추가
                break
    else:
        # SystemMessage가 없으면 새로 추가
        new_system_message = SystemMessage(
            content=feeling_str
        )
        msg_with_lt_memory.append(new_system_message)

    print(msg_with_lt_memory)
    return msg_with_lt_memory


def check_feeling_only_systemmsg(sysmsg, sensor_value):
    if sensor_value == 1:
        feeling_str = feeling_prompt + " " + feeling1
    elif sensor_value == 2:
        feeling_str = feeling_prompt + " " + feeling2
    elif sensor_value == 3:
        feeling_str = feeling_prompt + " " + feeling3
    else:  # sensor_value == 4
        feeling_str = feeling_prompt + " " + feeling4

    sysmsg.content += feeling_str  # 내용 추가

    print(sysmsg)
    return sysmsg


def calculate_score(input_string):
    # Use regex to find all ':' and '>'
    colons = re.findall(r':', input_string)
    semicolons = re.findall(r'>', input_string)

    # Calculate the score
    score = len(colons) * 60 - len(semicolons) * 60
    return score


def speak_text(text, lang="ko"):
    """
    입력된 문자열을 gTTS를 사용해 음성으로 출력합니다.

    Args:
        text (str): 음성으로 출력할 텍스트.
        lang (str): 텍스트의 언어. 기본값은 한국어("ko").
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            tts = gTTS(text, lang=lang)
            tts.save(temp_file.name)
            playsound(temp_file.name)
            os.unlink(temp_file.name)  # 재생 후 파일 삭제
    except Exception as e:
        print(f"오디오 출력 중 오류 발생: {e}")


def main_setting_tool(user_input):
    messages = [
    ]