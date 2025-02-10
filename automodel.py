from transformers import AutoModel
import torch
from transformers.pipelines import AutoTokenizer

# pipeline: 높은 수준의 추상화 제공. 토큰화, 모델 추론, 후처리가 자동으로 처리됨
# AutoModel: 낮은 수준의 제어 가능. 각 단계를 직접 구현해야 함

model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

text = "this is a test"
inputs = tokenizer(text, return_tensors="pt")
print(f"입력 텐서 크기:  {inputs['input_ids'].size()}")  # [batch_size, n_tokens]

inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    outputs = model(**inputs)
print("\noutputs", outputs)

print(outputs.last_hidden_state.size())
print(outputs.last_hidden_state[:, 0].size())
print(outputs.hidden_states)
