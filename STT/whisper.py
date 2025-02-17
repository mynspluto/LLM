import whisper
import pyaudio
import numpy as np
import time

# Whisper 모델 로드
model = whisper.load_model("large")

# 오디오 입력 설정
CHUNK = 10240  # 버퍼 크기
FORMAT = pyaudio.paInt16  # 16-bit 오디오 포맷
CHANNELS = 1  # 단일 채널 (모노)
RATE = 44100  # 샘플링 속도 (Whisper는 16kHz를 사용)

# PyAudio 초기화
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
)

print("Listening for real-time audio...")

try:
    buffer = []  # 오디오 데이터를 담는 버퍼
    start_time = time.time()

    while True:
        # 마이크에서 오디오 데이터 읽기
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, np.int16)
        buffer.append(audio_data)

        # 5초마다 오디오 데이터를 처리
        if time.time() - start_time > 5:
            # 버퍼 데이터를 하나로 합치기
            audio_chunk = np.concatenate(buffer, axis=0).astype(
                np.float32
            )  # float32로 변환
            buffer = []  # 버퍼 초기화
            start_time = time.time()

            # 음성 데이터를 16비트에서 -1.0 ~ 1.0 사이로 정규화
            audio_chunk = audio_chunk / 32768.0

            # Whisper로 텍스트 변환
            result = model.transcribe(
                audio_chunk, fp16=False, language="ko"
            )  # 한국어 예시
            print("Detected Text:", result["text"])

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
