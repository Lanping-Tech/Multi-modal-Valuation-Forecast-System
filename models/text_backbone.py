import torch
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import pipeline

class TextBackbone(torch.nn.Module):

    def __init__(self, pretrained_model_name='chinese-roberta-wwm-ext', output_dim=320, window_size=5):
        super(TextBackbone, self).__init__()
        self.extractor = AutoModel.from_pretrained(pretrained_model_name)
        self.fc = torch.nn.Linear(768, output_dim)
        self.output_dim = output_dim
        self.window_size = window_size


    def forward(self, x):
        input_ids = x['input_ids']
        attention_mask = x['attention_mask']
        token_type_ids = x['token_type_ids']
        outputs = torch.empty(0, self.window_size, self.output_dim).to(input_ids.device)
        for i in range(input_ids.shape[0]):
            input_ids_i = input_ids[i]
            attention_mask_i = attention_mask[i]
            token_type_ids_i = token_type_ids[i]
            text_input = {'input_ids':input_ids_i,'attention_mask':attention_mask_i,'token_type_ids':token_type_ids_i}
            outputs_i = self.extractor(**text_input)
            outputs_i = self.fc(outputs_i.pooler_output)
            outputs = torch.cat((outputs, outputs_i.unsqueeze(0)), dim=0)
        return outputs

if __name__ == '__main__':
    model = AutoModel.from_pretrained('chinese-roberta-wwm-ext')
    x = ['我爱学习', '我不爱学习']
    tokenizer = AutoTokenizer.from_pretrained('chinese-roberta-wwm-ext')
    x = tokenizer.encode_plus(x, max_length=512, padding='max_length', return_tensors='pt')
    print(x)
    y = model(**{'input_ids':x['input_ids'],'attention_mask':x['attention_mask'],'token_type_ids':x['token_type_ids']})
    print(y)