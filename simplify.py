from transformers import pipeline

# Hugging Face에서 KoBART 모델을 사용한 요약 파이프라인 불러오기
summarizer = pipeline("summarization", model="gogamza/kobart-base-v2")

# 요약할 한국어 텍스트
text = """
한국어 자연어 처리 분야에서 Hugging Face는 중요한 역할을 하고 있으며, 다양한 한국어 모델들이 
Hugging Face의 플랫폼을 통해 제공되고 있습니다. 특히 KoBART, KoT5와 같은 모델은 텍스트 생성, 
요약, 번역 등 다양한 작업을 지원하여 연구자와 개발자들에게 큰 도움이 되고 있습니다. 
Hugging Face는 오픈소스와 커뮤니티의 힘을 통해 NLP 분야에서의 혁신을 이루어가고 있습니다.
"""

# 텍스트 요약
summary = summarizer(text, max_length=50, min_length=25, do_sample=False)

# 결과 출력
print("Original Text:")
print(text)
print("\nSummarized Text:")
print(summary[0]["summary_text"])
