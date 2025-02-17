import speech_recognition as sr

WAKE_WORD = "통화"


def listen_and_recognize():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("한국어로 말하세요...")
        recognizer.adjust_for_ambient_noise(source)  # 주변 소음 보정
        try:
            audio = recognizer.listen(source)  # 음성 듣기
            # Google Speech-to-Text API를 사용하여 음성을 텍스트로 변환
            text = recognizer.recognize_google(audio, language="ko-KR")
            print(f"인식된 텍스트: {text}")

            # 웨이크 워드 감지
            if WAKE_WORD in text.lower():
                print("Wake word detected!")
                return True
        except sr.UnknownValueError:
            print("음성을 인식할 수 없습니다.")
        except sr.RequestError as e:
            print(f"Google Speech Recognition 서비스 오류: {e}")


# 실행
listen_and_recognize()
