import torch
from torch.utils.data import DataLoader
from datasets import MultiModalDataset
from data_preprocessing import load_data
from models.mts_backbone import MTSBackbone
from models.text_backbone import TextBackbone
from models.fusion import TCNT

import numpy as np

from tensorflow.keras.models import load_model
from tcn import TCN, tcn_full_summary
from tensorflow.keras.initializers import glorot_uniform


# Model settings
in_channels = 6
channels = 64
depth = 3
reduced_size = 160
out_channels = 320
kernel_size = 3
fusion_heads = 4
out_size = 1
window_size = 5
train_test_split = 0.8


gru_model_path = 'pretrained/gru_model.h5'
lstm_model_path = 'pretrained/lstm_model.h5'
tcn_model_path = 'pretrained/tcn_model.h5'

mts_data, text_data, means, stds = load_data('data/股价', 'data/文本数据', ['000001'], WINDOW_SIZE=window_size+1)

train_size = int(train_test_split * len(text_data))
train_mts_data, test_mts_data = mts_data[:train_size], mts_data[train_size:]
train_text_data, test_text_data = text_data[:train_size], text_data[train_size:]

x_test = test_mts_data[:, :window_size]
y_test = test_mts_data[:, window_size, 3:4]
y_test = (y_test * stds[3]) + means[3]

gru_model  = load_model(gru_model_path)
gru_pred = gru_model.predict(x_test)[:,0]
gru_pred = (gru_pred * stds[3]) + means[3]

lstm_model = load_model(lstm_model_path)
lstm_pred = lstm_model.predict(x_test)[:,0]
lstm_pred = (lstm_pred * stds[3]) + means[3]

tcn_model = load_model(tcn_model_path, custom_objects={'TCN': TCN, 'GlorotUniform': glorot_uniform()})
tcn_pred = tcn_model.predict(x_test)[:,0]
tcn_pred = (tcn_pred * stds[3]) + means[3]

test_dataset = MultiModalDataset(test_mts_data, test_text_data, window_size)
test_loader = DataLoader(dataset=test_dataset,
                                batch_size=8,
                                shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model
mts_model = MTSBackbone(in_channels, channels, depth, reduced_size, out_channels, kernel_size).to(device)
text_model = TextBackbone(output_dim=out_channels).to(device)
fusion_model = TCNT(out_channels, fusion_heads, out_size).to(device)

mts_model.load_state_dict(torch.load('pretrained/mts_model.pth'))
text_model.load_state_dict(torch.load('pretrained/text_model.pth'))
fusion_model.load_state_dict(torch.load('pretrained/fusion_model.pth'))

mts_model.eval()
text_model.eval()
fusion_model.eval()
test_loss = 0
tcnt_pred = []
with torch.no_grad():
    for batch_idx, (mts, text, label) in enumerate(test_loader):
        # print(batch_idx)
        mts, label = mts.to(device), label.to(device)
        input_ids,attention_mask,token_type_ids = text
        input_ids,attention_mask,token_type_ids = input_ids.to(device),attention_mask.to(device),token_type_ids.to(device)
        mts_output = mts_model(mts)
        text_output = text_model({'input_ids':input_ids,'attention_mask':attention_mask,'token_type_ids':token_type_ids})
        output = fusion_model(mts_output, text_output)
        tcnt_pred.append(output.cpu().numpy())


tcnt_pred = np.concatenate(tcnt_pred)[:,0]
tcnt_pred = (tcnt_pred * stds[3]) + means[3]

result = [y_test, gru_pred, lstm_pred, tcn_pred, tcnt_pred]
name = ['ground_truth', 'gru', 'lstm', 'tcn', 'tcnt']

import os
import matplotlib.pyplot as plt

color = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
for i in range(len(result)):
    plt.plot(list(range(1, len(result[i])+1)), result[i], color[i], label=name[i])

plt.legend()
plt.xlabel('Time steps')
plt.ylabel('Stock price')
plt.grid(linestyle='--')
plt.savefig('predict_result.png', dpi=500, bbox_inches = 'tight')



