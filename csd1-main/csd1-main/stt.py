import speech_recognition as sr
import numpy as np
import time
from collections import deque

def recognize_speech(device_index=4, volume_threshold=3, no_sound_limit=2, language="ko-KR"):
    """
    음성을 인식하고 최종 결과를 문자열로 반환하는 함수.

    Args:
        device_index (int): 마이크 디바이스 인덱스 (`arecord -l`로 확인 가능)
        volume_threshold (float): 음량 임계값
        no_sound_limit (int): 소리가 없을 경우 종료할 시간 (초)
        language (str): 음성 인식 언어 (기본값은 한국어)

    Returns:
        str: 인식된 텍스트
    """
    recognizer = sr.Recognizer()
    accumulated_text = ""  # 텍스트 버퍼

    with sr.Microphone(device_index=device_index) as microphone_source:
        print("준비 중...")
        recognizer.adjust_for_ambient_noise(microphone_source)
        print("배경 소음 기준 조정 완료")

        no_sound_start = None  # 소리 입력이 없어진 시간
        print("소리 듣는 중...")

        while True:
            # 오디오 데이터 읽기
            audio_data = recognizer.listen(microphone_source, phrase_time_limit=2)  # 3초씩 녹음 후 인식
            audio_samples = np.frombuffer(audio_data.frame_data, np.int16)
            average_volume = np.linalg.norm(audio_samples) / len(audio_samples)

            if average_volume > volume_threshold:
                # 소리 감지됨 - 음성 인식 수행
                print("소리 감지됨, 인식 중...")

                try:
                    # Google API를 사용한 음성 인식
                    recognized_text = recognizer.recognize_google(audio_data, language=language)
                    accumulated_text += recognized_text + " "  # 인식된 텍스트 이어 붙이기
                    print("누적된 텍스트:", accumulated_text)

                    # 소리가 감지될 때마다 no_sound_start 리셋
                    no_sound_start = None

                except sr.UnknownValueError:
                    print("인식 실패")
                except sr.RequestError as error:
                    print("Google API 요청 실패:", error)
            else:
                # 소리가 감지되지 않으면 no_sound_start 카운터 시작
                if no_sound_start is None:
                    no_sound_start = time.time()

                # 소리가 없던 시간이 no_sound_limit를 넘으면 아예 종료
                elif time.time() - no_sound_start > no_sound_limit:
                    print("더이상 소리가 감지되지 않음.")
                    break

    # 최종 출력
    final_output = accumulated_text.strip()  # 텍스트 양 끝 공백 제거
    return final_output

def recognize_speech_new(threshold, listen_duration=1, terminate_duration=1, language="ko-KR"):
    """
    계산된 임계값을 기반으로 소리를 듣고, 임계값 이하 소리가 지속되면 종료.

    Args:
        threshold (float): 에너지 임계값.
        listen_duration (float): 녹음 시간 (초).
        terminate_duration (float): 임계값 이하 지속 시간 (초).
        language (str): 음성 인식 언어.

    Returns:
        str: 인식된 텍스트 결과.
    """
    start_time = time.time()

    recognizer = sr.Recognizer()
    accumulated_text = ""  # 누적된 텍스트
    no_sound_start = None  # 임계값 이하가 시작된 시간
    capturing_audio = False  # 소리를 캡처 중인지 여부
    audio_chunks = []  # 임계값 초과 구간의 오디오 데이터 저장

    with sr.Microphone() as source:
        print("소리 듣는 중...")

        while True:
            print(f"Listening for {listen_duration} seconds...")
            audio_data = recognizer.listen(source, phrase_time_limit=listen_duration)

            # 오디오 데이터의 에너지 계산
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            energy = recognizer.energy_threshold

            print(f"[실시간 Energy] {energy:.2f} (Threshold: {threshold})")

            if energy > threshold:
                # 임계값 초과 시 오디오 캡처 시작
                print("소리 감지됨, 캡처 중...")
                capturing_audio = True
                audio_chunks.append(audio_data)  # 현재 오디오 데이터를 저장
                no_sound_start = None  # 임계값 이하 시간 초기화

            else:
                # 임계값 이하일 때 처리
                if capturing_audio:
                    print(f"Threshold 이하의 소리 감지됨: {energy:.2f}")
                    if no_sound_start is None:
                        no_sound_start = time.time()
                    elif time.time() - no_sound_start >= terminate_duration:
                        print("임계값 이하 소리가 1초 이상 지속됨. 캡처 종료 및 처리 시작.")
                        
                        # 저장된 오디오 데이터를 하나로 결합
                        combined_audio = sr.AudioData(
                            b"".join(chunk.get_raw_data() for chunk in audio_chunks),
                            audio_chunks[0].sample_rate,
                            audio_chunks[0].sample_width,
                        )

                        # Google Speech-to-Text API로 전송
                        try:
                            recognized_text = recognizer.recognize_google(combined_audio, language=language)
                            accumulated_text += recognized_text + " "
                            print(f"누적된 텍스트: {accumulated_text}")
                        except sr.UnknownValueError:
                            print("음성 인식 실패.")
                        except sr.RequestError as e:
                            print(f"Google API 오류: {e}")
                        
                        # 데이터 초기화 및 종료
                        audio_chunks.clear()
                        capturing_audio = False
                        break  # 종료 조건
                else:
                    print(f"Threshold 이하의 소리 무시 중: {energy:.2f}")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"총 걸린 시간: {elapsed_time:.2f}초")
    # 누적된 텍스트 반환
    final_output = accumulated_text.strip()  # 텍스트 양 끝 공백 제거    
    return final_output

def recognize_speech_fivesec(threshold=800, language="ko-KR", device_index=3, pre_record_duration=2):
    start_time = time.time() #!!!!!!!!시작 시간
    recognizer = sr.Recognizer()

    # 오디오 버퍼 설정 (예: 0.5초짜리 청크로 pre_record_duration 만큼 저장)
    buffer_duration = 0.5  # 각 청크의 길이 (초)
    max_chunks = int(pre_record_duration / buffer_duration)
    audio_buffer = deque(maxlen=max_chunks)

    with sr.Microphone(device_index=device_index) as source:
        print(f"소리 듣는 중... (Device Index: {device_index})")
        while True:
            # 짧은 오디오 청크 녹음
            audio_chunk = recognizer.record(source, duration=buffer_duration)
            audio_buffer.append(audio_chunk)

            # 주변 소음으로 에너지 임계값 조정
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            current_threshold = recognizer.energy_threshold
            print(f"[현재 에너지 임계값] {current_threshold:.2f} (설정된 Threshold: {threshold})")

            # 에너지 수준 비교
            if current_threshold > threshold:
                print(f"에너지 임계값 초과 감지됨: {current_threshold:.2f}")
                # 에너지 임계값 초과 시 추가로 5초간 녹음
                print("2초 동안 녹음 시작...")

                # 추가로 녹음할 시간 계산
                remaining_duration = 2.0
                audio_data_list = list(audio_buffer)  # 버퍼에 저장된 오디오 청크들

                while remaining_duration > 0:
                    duration = min(buffer_duration, remaining_duration)
                    audio_chunk = recognizer.record(source, duration=duration)
                    audio_data_list.append(audio_chunk)
                    remaining_duration -= duration

                # 모든 오디오 청크를 하나로 합치기
                combined_audio_data = sr.AudioData(
                    b"".join([chunk.get_raw_data() for chunk in audio_data_list]),
                    source.SAMPLE_RATE,
                    source.SAMPLE_WIDTH
                )

                # 음성 인식
                try:
                    recognized_text = recognizer.recognize_google(combined_audio_data, language=language)
                    print(f"인식된 텍스트: {recognized_text}")
                    end_time = time.time() #!!!!!!!!종료 시간
                    elapsed_time = end_time - start_time #!!!!!!!!시간계산
                    print(f"총 걸린 시간: {elapsed_time:.2f}초") #!!!!!!!!!!시간 출력
                    return recognized_text
                except sr.UnknownValueError:
                    print("음성 인식 실패.")
                    return ""
                except sr.RequestError as e:
                    print(f"Google API 오류: {e}")
                    return ""
            else:
                print(f"에너지 임계값 이하: {current_threshold:.2f}")
                time.sleep(0.1)  # 잠시 대기 후 다시 감지

# user_input = recognize_speech_fivesec(threshold=800, language="ko-KR", device_index=3, pre_record_duration=2)

