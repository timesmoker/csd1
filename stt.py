import speech_recognition as sr
import numpy as np
import time
from collections import deque

def recognize_speech_fivesec(threshold=800, language="ko-KR", device_index=3, pre_record_duration=2):
    start_time = time.time()  # 시작 시간
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
                    end_time = time.time()  # 종료 시간
                    elapsed_time = end_time - start_time  # 시간 계산
                    print(f"총 걸린 시간: {elapsed_time:.2f}초")  # 시간 출력
                    return recognized_text
                except sr.UnknownValueError:
                    print("음성 인식 실패. 다시 시도합니다.")
                    return recognize_speech_fivesec(threshold, language, device_index, pre_record_duration)
                except sr.RequestError as e:
                    print(f"Google API 오류: {e}. 다시 시도합니다.")
                    return recognize_speech_fivesec(threshold, language, device_index, pre_record_duration)
            else:
                print(f"에너지 임계값 이하: {current_threshold:.2f}")
                time.sleep(0.1)  # 잠시 대기 후 다시 감지
