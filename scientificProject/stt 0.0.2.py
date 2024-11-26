import speech_recognition as sr
import numpy as np
import time

recognizer = sr.Recognizer()
VOLUME_THRESHOLD = 18  # 음량 임계값 (실제 환경에 맞게 조정 필요)
NO_SOUND_LIMIT = 5  # 소리가 없을 경우 종료할 시간 (초)
accumulated_text = ""  # 텍스트 버퍼

with sr.Microphone(device_index=3) as microphone_source:
    print("준비 중...")
    recognizer.adjust_for_ambient_noise(microphone_source)
    print("배경 소음 기준 조정 완료")

    no_sound_start = None  # 소리 입력이 없어진 시간

    print("소리 듣는중......")

    while True:
        # 오디오 데이터 읽기
        audio_data = recognizer.listen(microphone_source, phrase_time_limit=3)  # 3초씩 녹음 후 인식
        audio_samples = np.frombuffer(audio_data.frame_data, np.int16)
        average_volume = np.linalg.norm(audio_samples) / len(audio_samples)

        if average_volume > VOLUME_THRESHOLD:
            # 소리 감지됨 - 음성 인식 수행
            print("소리 감지됨, 인식 중...")

            try:
                # Google API를 사용한 음성 인식
                recognized_text = recognizer.recognize_google(audio_data, language='ko-KR')
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

            # 소리가 없던 시간이 NO_SOUND_LIMIT를 넘으면 아예 종료
            elif time.time() - no_sound_start > NO_SOUND_LIMIT:
                print("더이상 소리가 감지 되지 않음.")
                break
