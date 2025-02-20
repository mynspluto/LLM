import pyaudio
import numpy as np
import wave
import time
from datetime import datetime


class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.chunk = 2048
        self.stream = None
        self.audio_data_list = []

    def start_recording(self):
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )

            print("녹음 시작... (Ctrl+C로 중지)")

            while True:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    # 원본 바이너리 데이터를 저장 (WAV 파일 저장용)
                    self.audio_data_list.append(data)

                    # 디스플레이용 데이터 변환
                    audio_chunk = np.frombuffer(data, dtype=np.int16)

                    # 현재 청크 데이터 출력
                    print("\r현재 청크:", end=" ")
                    print("[", end="")
                    for i, value in enumerate(audio_chunk[:5]):
                        print(f"{value:6d}", end="")
                        if i < 4:
                            print(", ", end="")
                    print("]", end="")

                except OSError as e:
                    print(f"\n스트림 읽기 오류: {e}")
                    continue

        except KeyboardInterrupt:
            print("\n녹음 중지")

            # 녹음 파일 저장
            self.save_recording()

            # 통계 출력
            all_data = np.frombuffer(b"".join(self.audio_data_list), dtype=np.int16)
            print("\n녹음 통계:")
            print(f"총 녹음 시간: {len(all_data)/self.rate:.2f}초")
            print(f"총 샘플 수: {len(all_data)}")
            print(f"평균 음량: {np.mean(np.abs(all_data)):.2f}")
            print(f"최대 음량: {np.max(np.abs(all_data))}")

        finally:
            if self.stream is not None:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except OSError as e:
                    print(f"\n스트림 종료 오류: {e}")

    def save_recording(self):
        if not self.audio_data_list:
            print("저장할 녹음 데이터가 없습니다.")
            return

        # 현재 시간을 파일명에 포함
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

        try:
            # WAV 파일로 저장
            wf = wave.open(filename, "wb")
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(self.audio_data_list))
            wf.close()

            print(f"\n녹음이 {filename}에 저장되었습니다.")

            # 파일 크기 확인
            import os

            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB 단위
            print(f"파일 크기: {file_size:.2f} MB")

        except Exception as e:
            print(f"\n파일 저장 중 오류 발생: {e}")

    def __del__(self):
        self.audio.terminate()


if __name__ == "__main__":
    recorder = AudioRecorder()
    recorder.start_recording()
