import os
from openai import OpenAI

api_key_from_os = os.getenv("OPENAI_API_KEY")  # 환경변수에서 API 키 가져오기

client = OpenAI(api_key=api_key_from_os)
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[{"role": "user", "content": "write a haiku about ai"}],
)
print(completion.choices[0].message)
