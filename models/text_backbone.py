import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import pipeline

class TextFeatureExtractor(torch.nn.Module):

    def __init__(self, pretrained_model_name='hfl/chinese-roberta-wwm-ext', output_dim=320):
        super(TextFeatureExtractor, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained_model_name)
        self.fc = torch.nn.Linear(768, output_dim)


    def forward(self, x):
        outputs = self.extractor(**x)
        out = self.fc(outputs.pooler_output)
        return out
