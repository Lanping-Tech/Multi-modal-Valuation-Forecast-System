import os
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat

from models.text_backbone import TextBackbone
from models.mts_backbone import MTSBackbone

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class FusionAttention(nn.Module):
    def __init__(self, dim, heads = 4, dropout = 0.):
        super().__init__()

        self.heads = heads
        self.scale = dim ** -0.5

        self.attend = nn.Softmax(dim = -1)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, mts_features, text_features):
        q = rearrange(text_features, 'b n (h d) -> b h n d', h = self.heads)
        v = rearrange(mts_features, 'b n (h d) -> b h n d', h = self.heads)

        dots = torch.matmul(q, v.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TCNT(nn.Module):
    def __init__(self, out_channels=320, fusion_heads=4, out_size=1, **kwargs):
        super(TCNT, self).__init__()

        self.fusion = FusionAttention(out_channels, fusion_heads)
        self.fusion_norm = nn.LayerNorm(out_channels)

        self.ff = FeedForward(out_channels, out_channels*2)
        self.ff_norm = nn.LayerNorm(out_channels)

        self.to_out_lstm = nn.LSTM(out_channels, out_channels, batch_first=True, bidirectional=True)
        self.to_out_norm = nn.LayerNorm(out_channels)

        self.to_out_linear = nn.Linear(out_channels*2, out_size)

    def forward(self, mts_features, text_features):
        mts_features = self.fusion_norm(mts_features)
        text_features = self.fusion_norm(text_features)
        fusion_features = self.fusion(mts_features, text_features)

        fusion_features = self.ff_norm(fusion_features)
        fusion_features = self.ff(fusion_features)

        fusion_features = self.to_out_norm(fusion_features)
        _, out_feature = self.to_out_lstm(fusion_features)

        out = out_feature[0].permute(1, 0, 2).contiguous().view(mts_features.size(0),-1)

        out = self.to_out_linear(out)

        return out