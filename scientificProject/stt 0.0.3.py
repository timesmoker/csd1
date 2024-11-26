import speech_recognition as sr
import numpy as np
from vosk import Model, KaldiRecognizer
import wave

VOLUME_THRESHOLD = 18  # 음량 임계값
NO_SOUND_LIMIT = 5  # 소리가 없을 경우 종료할 시간 (초)
accumulated_text = ""  # 텍스트 버퍼

# Vosk 한국어 모델 경로 설정
model_path = "model/vosk-model-small-ko-0.22"  # 다운로드한 모델 파일이 위치한 경로
model = Model(model_path)

# 오디오 파일 경로 설정
audio_file_path = "path/to/your_audio_file.wav"

# 오디오 파일을 읽어 인식하는 함수
def recognize_audio_file(file_path):
    global accumulated_text

    # 오디오 파일 열기
    with wave.open(file_path, "rb") as audio_file:
        # Vosk 인식기 초기화
        vosk_recognizer = KaldiRecognizer(model, audio_file.getframerate())

        # 오디오 파일에서 프레임 단위로 읽어서 Vosk로 인식
        while True:
            data = audio_file.readframes(4000)  # 한 번에 4000프레임씩 읽기
            if len(data) == 0:
                break  # 데이터가 더 이상 없으면 종료

            if vosk_recognizer.AcceptWaveform(data):
                recognized_text = vosk_recognizer.Result()
                accumulated_text += recognized_text + " "
                print("누적된 텍스트:", accumulated_text)

        # 마지막 결과 출력
        final_result = vosk_recognizer.FinalResult()
        accumulated_text += final_result + " "
        print("최종 텍스트:", accumulated_text)

# 함수 호출
recognize_audio_file(audio_file_path)
