import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import pipeline

class TextBackbone(torch.nn.Module):

    def __init__(self, pretrained_model_name='chinese-roberta-wwm-ext', output_dim=320):
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained_model_name)
        self.fc = torch.nn.Linear(768, output_dim)
        self.output_dim = output_dim


    def forward(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        token_type_ids = x['token_type_ids']
        outputs = torch.empty(0, 5, self.output_dim).to(input_ids.device)
        for i in range(input_ids.shape[0]):
            input_ids_i = input_ids[i]
            attention_mask_i = attention_mask[i]
            token_type_ids_i = token_type_ids[i]
            text_input = {'input_ids':input_ids_i,'attention_mask':attention_mask_i,'token_type_ids':token_type_ids_i}
            print(input_ids_i.shape)
            outputs_i = self.extractor(**text_input)
            outputs_i = self.fc(outputs_i)
            outputs = torch.cat((outputs, outputs_i.unsqueeze(0)), dim=0)
        return outputs

if __name__ == '__main__':
    model = AutoModel.from_pretrained('chinese-roberta-wwm-ext')
    x = ['我爱学习']
    tokenizer = AutoTokenizer.from_pretrained('chinese-roberta-wwm-ext')
    x = tokenizer(x, add_special_tokens=True, max_length=512, padding=True, return_tensors='pt')
    print(x)
    y = model(**x)
    print(y)