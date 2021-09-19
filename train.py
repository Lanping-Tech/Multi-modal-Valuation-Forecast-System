import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl

from model import LitAutoEncoder

dataset = MNIST(os.getcwd(), download=True, transform=transforms.ToTensor())
train_loader = DataLoader(dataset)

# init model
autoencoder = LitAutoEncoder()

trainer = pl.Trainer()
trainer.fit(autoencoder, train_loader)