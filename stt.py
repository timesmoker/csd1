import speech_recognition as sr
import numpy as np
import time

def recognize_speech(device_index=3, volume_threshold=3, no_sound_limit=2, language="ko-KR"):
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

