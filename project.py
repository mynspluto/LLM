import queue
import sys
import json
import pyaudio
import vosk
import re
import os

# 키워드 리스트
KEYWORDS = ["안녕", "제트슨", "음성 인식", "시작"]

# 현재 실행 중인 Python 파일의 디렉토리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-ko-0.22")

# Vosk 모델 로드
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, 16000)

# 오디오 입력 버퍼
audio_queue = queue.Queue()


def audio_callback(in_data, frame_count, time_info, status):
    """오디오 콜백 함수"""
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)


# PyAudio 설정
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=8000,
    stream_callback=audio_callback,
)

print("🎙️ 음성 인식 시작... (Ctrl+C로 종료)")
stream.start_stream()

try:
    while True:
        if not audio_queue.empty():
            data = audio_queue.get()
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "")
                if text:
                    print(f"🗣️ 인식된 텍스트: {text}")

                    # 키워드 검색
                    for keyword in KEYWORDS:
                        if re.search(rf"\b{keyword}\b", text):
                            print(f"🔍 키워드 '{keyword}' 감지됨!")

except KeyboardInterrupt:
    print("\n🛑 음성 인식 종료")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
