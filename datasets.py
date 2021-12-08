import torch
import torch.utils.data as data
import numpy as np
import os


class MultiModalDataset(data.Dataset):
    def __init__(self, mts_data, text_data):
        self.mts_data = mts_data
        self.text_data = text_data

    def __getitem__(self, index):
        mts, text = self.mts_data[index], self.mts_data[index]
        return mts, text, mts[-1]

    def __len__(self):
        return len(self.mts_data)