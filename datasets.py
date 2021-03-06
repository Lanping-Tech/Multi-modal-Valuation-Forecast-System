import torch
import torch.utils.data as data
import numpy as np
import os
from data_preprocessing import load_data
from transformers import AutoTokenizer


class MultiModalDataset(data.Dataset):
    def __init__(self, mts_data, text_data, window_size=5):
        self.window_size = window_size
        self.mts_data = mts_data
        self.text_data = text_data
        self.tokenizer = AutoTokenizer.from_pretrained('chinese-roberta-wwm-ext')
        # print(len(self.mts_data), len(self.text_data))

    def __getitem__(self, index):
        ts, text = self.mts_data[index], self.text_data[index]
        mts = ts[:self.window_size, :]
        mts = torch.tensor(mts, dtype=torch.float)
        label = ts[self.window_size, 3:4]
        label = torch.tensor(label, dtype=torch.float)
        content = []
        # print(len(text))
        for i in range(self.window_size):
            candidated_text = text[i]
            seleted_id = np.random.randint(len(candidated_text))
            content.append(candidated_text[seleted_id])
            
        input_ids_list = torch.empty(0, 128, dtype=torch.long)
        attention_mask_list = torch.empty(0, 128, dtype=torch.long)
        token_type_ids_list = torch.empty(0, 128, dtype=torch.long)

        for i in range(len(content)):
            content_encoding = self.tokenizer(content[i], add_special_tokens=True, max_length=128, padding='max_length', return_tensors='pt')
            input_ids = content_encoding['input_ids']
            attention_mask = content_encoding['attention_mask']
            token_type_ids = content_encoding['token_type_ids']
            input_ids_list = torch.cat((input_ids_list, input_ids), dim=0)
            attention_mask_list = torch.cat((attention_mask_list, attention_mask), dim=0)
            token_type_ids_list = torch.cat((token_type_ids_list, token_type_ids), dim=0)
        return mts, (input_ids_list, attention_mask_list, token_type_ids_list), label

    def __len__(self):
        return len(self.mts_data)


class MultiModalDataset_plus(data.Dataset):
    def __init__(self, mts_path, text_path, stock_ids, window_size=5):
        self.window_size = window_size
        self.mts_data, self.text_data, _, _ = load_data(mts_path, text_path, stock_ids, WINDOW_SIZE=window_size+1)
        self.tokenizer = AutoTokenizer.from_pretrained('chinese-roberta-wwm-ext')
        # print(len(self.mts_data), len(self.text_data))

    def __getitem__(self, index):
        ts, text = self.mts_data[index], self.text_data[index]
        mts = ts[:self.window_size, :]
        mts = torch.tensor(mts, dtype=torch.float)
        label = ts[self.window_size, 3:4]
        label = torch.tensor(label, dtype=torch.float)
        content = []
        # print(len(text))
        for i in range(self.window_size):
            candidated_text = text[i]
            seleted_id = np.random.randint(len(candidated_text))
            content.append(candidated_text[seleted_id])
            
        input_ids_list = torch.empty(0, 128, dtype=torch.long)
        attention_mask_list = torch.empty(0, 128, dtype=torch.long)
        token_type_ids_list = torch.empty(0, 128, dtype=torch.long)

        for i in range(len(content)):
            content_encoding = self.tokenizer(content[i], add_special_tokens=True, max_length=128, padding='max_length', return_tensors='pt')
            input_ids = content_encoding['input_ids']
            attention_mask = content_encoding['attention_mask']
            token_type_ids = content_encoding['token_type_ids']
            input_ids_list = torch.cat((input_ids_list, input_ids), dim=0)
            attention_mask_list = torch.cat((attention_mask_list, attention_mask), dim=0)
            token_type_ids_list = torch.cat((token_type_ids_list, token_type_ids), dim=0)
        return mts, (input_ids_list, attention_mask_list, token_type_ids_list), label

    def __len__(self):
        return len(self.mts_data)


if __name__ == '__main__':
    data = MultiModalDataset('data/??????', 'data/????????????', ['000001'])
    for i in range(64):
        print(i, data[i][0].shape)
        print()