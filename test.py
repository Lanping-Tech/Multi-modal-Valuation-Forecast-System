import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import pipeline

tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext")
model = AutoModel.from_pretrained("chinese-roberta-wwm-ext")

inputs = tokenizer(
            ['文本1','文本2'],
            add_special_tokens=True,
            return_tensors="pt",
            padding=True,
        )
# print(inputs)
outputs = model(**inputs)

print(outputs.pooler_output.shape)
