import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from models.Informer import Model as Informer
from models.TCN import TemporalConvNet
from models.modules.HybridScaleFeatureExtraction import HybridScaleFeatureExtraction

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# 动态稀疏注意力机制
class DynamicSparseAttention(nn.Module):
    def __init__(self, embed_size, heads, sparsity):
        super(DynamicSparseAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.sparsity = sparsity
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

        if sparsity < 0 or sparsity >= 1:
            raise ValueError("sparsity must be between 0 and 1")

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.view(N, value_len, self.heads, self.embed_size // self.heads)
        keys = keys.view(N, key_len, self.heads, self.embed_size // self.heads)
        queries = queries.view(N, query_len, self.heads, self.embed_size // self.heads)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.nn.functional.softmax(energy, dim=3)

        k = int((1 - self.sparsity) * attention.shape[-1])
        if k <= 0:
            raise ValueError("k value is non-positive, check sparsity and attention dimensions.")
        topk, _ = attention.topk(k, dim=-1)
        min_topk = topk[:, :, :, -1].unsqueeze(-1).expand_as(attention)
        sparse_attention = torch.where(attention < min_topk, torch.zeros_like(attention), attention)

        out = torch.einsum("nhql,nlhd->nqhd", [sparse_attention, values]).reshape(
            N, query_len, self.embed_size
        )
        out = self.fc_out(out)
        return out

# 混合多尺度稀疏注意力时序卷积网络
class HSFE_GRU_Informer_DSA(nn.Module):
    def __init__(self, config,input_size, tcn_channels, gru_hidden_size, embed_size, heads, sparsity, output_size=1, kernel_size=2, dropout=0.2):
        super(HSFE_GRU_Informer_DSA, self).__init__()
        self.hybrid_scale = HybridScaleFeatureExtraction(input_size)
        self.tcn = TemporalConvNet(input_size, tcn_channels, kernel_size=kernel_size, dropout=dropout)
        self.gru = nn.GRU(tcn_channels[-1], gru_hidden_size, bidirectional=True, batch_first=True)
        self.informer = Informer(config)
        self.attention = DynamicSparseAttention(gru_hidden_size * 2, heads, sparsity)
        self.fc = nn.Linear(gru_hidden_size * 2, output_size)

    def forward(self, x):
        hybrid_out = self.hybrid_scale(x)
        hybrid_out = self.informer(hybrid_out)
        tcn_out = self.tcn(hybrid_out.permute(0, 2, 1))
        gru_out, _ = self.gru(tcn_out.permute(0, 2, 1))
        attn_out = self.attention(gru_out, gru_out, gru_out)
        out = self.fc(attn_out[:, -1, :])
        return out
