import pandas as pd
import torch
from torch.nn import functional as F

text = "Tokenizing text is a core task of NLP."

# 문자 토큰화
tokenized_text = list(text)
print("tokenized_text\n", tokenized_text)

token2idx = {ch: idx for idx, ch in enumerate(sorted(set(tokenized_text)))}
print("\ntoken2idx\n", token2idx)

input_ids = [token2idx[token] for token in tokenized_text]
print("\ninput_ids\n", input_ids)

categorical_df = pd.DataFrame(
    {"Name": ["Bumblebee", "Optimus Prime", "Megatron"], "Label ID": [0, 1, 2]}
)

# 이름 사이에 순서가 만들어지는 문제 발생
print("\ncategorical_df\n", categorical_df)

# 위의 문제 해결, 범주마다 새 열을 만들어 이름이 범주에 해당하면 1 아니면 0
# [1, 0, 0], [0, 1, 0], [0, 0, 1] 벡터로 변환[1, 1, 0]의 경우 범블비와 옵티머스 프라임이 동시에 존재한다는 의미
# 원소 사이에 순서가 생기는 비슷한 문제 발생
print("\ncategorical_df vector\n", pd.get_dummies(categorical_df["Name"]))

input_ids = torch.tensor(input_ids)
one_hot_encodings = F.one_hot(input_ids, num_classes=len(token2idx))

# 38(text.length) 개의 입력 토큰 가각에 20(token2idx.key.length) 차원의 원-핫 벡터 생성
print("\none_hot_encodings.shape\n", one_hot_encodings.shape)
