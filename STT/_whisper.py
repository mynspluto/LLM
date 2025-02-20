import whisper
import os

# 맥 음성 메모로 만든 파일
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "../audio.m4a")

# https://github.com/openai/whisper
model = whisper.load_model("medium")
result = model.transcribe(file_path)
print(result["text"])
