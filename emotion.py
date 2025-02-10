from transformers import pipeline

model_id = "hun3359/klue-bert-base-sentiment"
model_pipeline = pipeline("text-classification", model=model_id)

# 분석할 텍스트
text = "오늘은 기분이 너무 좋고 행복해요!"
text2 = "우울하고 불안하다"

# 감정 분석 결과
result = model_pipeline(text)
print(result)

result = model_pipeline(text2)
print(result)
