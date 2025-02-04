import pandas as pd
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

text = "Tokenizing text is a core task of NLP."

# 문자 토큰화
tokenized_text = list(text)
print("tokenized_text", tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print("\ntoken2idx", token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
print("\ninput_ids", input_ids)

categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0, 1, 2]}
)

# 이름 사이에 순서가 만들어지는 문제 발생
print("\ncategorical_df", categorical_df)

# 위의 문제 해결, 범주마다 새 열을 만들어 이름이 범주에 해당하면 1 아니면 0
# [1, 0, 0], [0, 1, 0], [0, 0, 1] 벡터로 변환[1, 1, 0]의 경우 범블비와 옵티머스 프라임이 동시에 존재한다는 의미
print("\ncategorical_df vector", pd.get_dummies(categorical_df["Name"]))

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))

# 38(text.length) 개의 입력 토큰 각각에 20(token2idx.key.length) 차원의 원-핫 벡터 생성
print("\none_hot_encodings.shape", one_hot_encodings.shape)
print(f"토큰: {tokenized_text[0]}")
print(f"토큰: {input_ids[0]}")
print(f"토큰: {one_hot_encodings[0]}")

# 단어 토큰화
tokenized_text = text.split()
print("\ntokenized_text", tokenized_text)

# 부분 단어 토큰화 DisitilBERT 토크나이저 로드
model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
encoded_text = tokenizer(text)
print("\nencoded_text", encoded_text)

tokens = tokenizer.convert_ids_to_tokens(encoded_text.input_ids)
print("\ntokens", tokens)
print("\nstring", tokenizer.convert_tokens_to_string(tokens))
