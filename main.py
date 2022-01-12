import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from datasets import MultiModalDataset
from models.mts_backbone import MTSBackbone
from models.text_backbone import TextBackbone
from models.fusion import TCNT

from transformers import AdamW



def train(models, device, train_loader, optimizers, epoch):
    mts_model, text_model, fusion_model = models
    optimizer_1, optimizer_2 = optimizers
    mts_model.train()
    text_model.train()
    fusion_model.train()
    for batch_idx, (mts, text, label) in enumerate(train_loader):
        mts, label = mts.to(device), label.to(device)
        input_ids,attention_mask,token_type_ids = text
        input_ids,attention_mask,token_type_ids = input_ids.to(device),attention_mask.to(device),token_type_ids.to(device)
        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(token_type_ids.shape)
        # print()
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        mts_output = mts_model(mts)
        text_output = text_model({'input_ids':input_ids,'attention_mask':attention_mask,'token_type_ids':token_type_ids})
        output = fusion_model(mts_output, text_output)

        loss = F.mse_loss(output, label)
        loss.backward()
        optimizer_1.step()
        optimizer_2.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(mts), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(models, device, test_loader):
    mts_model, text_model, fusion_model = models
    mts_model.eval()
    text_model.eval()
    fusion_model.eval()
    test_loss = 0
    with torch.no_grad():
        for mts, text, label in test_loader:
            mts, label = mts.to(device), label.to(device)
            input_ids,attention_mask,token_type_ids = text
            input_ids,attention_mask,token_type_ids = input_ids.to(device),attention_mask.to(device),token_type_ids.to(device)
            mts_output = mts_model(mts)
            text_output = text_model({'input_ids':input_ids,'attention_mask':attention_mask,'token_type_ids':token_type_ids})
            output = fusion_model(mts_output, text_output)
            test_loss += F.mse_loss(output, label, reduction='sum').item()  # sum up batch loss
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # correct += pred.eq(label.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(
        test_loss))

def main():
    # Data settings
    in_channels = 6
    channels = 64
    depth = 3
    reduced_size = 160
    out_channels = 320
    kernel_size = 3

    # Model settings
    fusion_heads = 4
    out_size = 1
    

    # Training settings
    batch_size = 64
    epochs = 10

    # Dataset
    train_dataset = MultiModalDataset('data/股价', 'data/文本数据', ['000001'])#, '000858', '300003', '300014'])

    test_dataset = MultiModalDataset('data/股价', 'data/文本数据', ['600104'])#, '600887', '600900'])

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)

    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,

                                shuffle=False)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    mts_model = MTSBackbone(in_channels, channels, depth, reduced_size, out_channels, kernel_size).to(device)
    text_model = TextBackbone(output_dim=out_channels).to(device)
    fusion_model = TCNT(out_channels, fusion_heads, out_size).to(device)

    # Optimizer
    optimizer_1 = torch.optim.Adam(list(mts_model.parameters()) + list(fusion_model.parameters()), lr=0.001)
    optimizer_2 = AdamW(text_model.parameters(),lr=2e-5, eps=1e-8)

    models = [mts_model, text_model, fusion_model]
    optimizers = [optimizer_1, optimizer_2]

    for epoch in range(1, epochs + 1):
        train(models, device, train_loader, optimizers, epoch)
        # test(models, device, test_loader)

if __name__ == '__main__':
    main()
