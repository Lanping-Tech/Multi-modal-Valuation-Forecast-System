import torch
from transformers import AutoModel

class TextFeatureExtractor(torch.nn.Module):

    def __init__(self, pretrained_model_name='hfl/chinese-roberta-wwm-ext', output_dim=320):
        super(TextFeatureExtractor, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained_model_name)
        self.fc = torch.nn.Linear(768, output_dim)


    def forward(self, x):
        _, pooled = self.extractor(**x)
        out = self.fc(pooled)
        return out

# tokenizer = AutoTokenizer.from_pretrained("chinese-roberta-wwm-ext")
# model = AutoModel.from_pretrained("chinese-roberta-wwm-ext")

# inputs = tokenizer(
#             ['文本1','文本2'],
#             add_special_tokens=True,
#             return_tensors="pt",
#             padding=True,
#         )
# print(inputs)
# x,y = model(inputs['input_ids'])

# print(x.shape)
# print(y.shape)
