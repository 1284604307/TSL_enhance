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
from models.modules.DynamicSparseAttention import DynamicSparseAttention
from models.modules.HybridScaleFeatureExtraction import HybridScaleFeatureExtraction

# 混合多尺度稀疏注意力时序卷积网络
class Model(nn.Module):
    def __init__(self, config, heads=2, sparsity=0, output_size=1, kernel_size=2, dropout=0.2):
        super(Model, self).__init__()
        self.input_size = config.enc_in
        self.tcn_channels = config.num_channels
        self.gru_hidden_size = config.p_hidden_layers
        self.heads = heads

        self.hybrid_scale = HybridScaleFeatureExtraction(self.input_size)
        self.tcn = TemporalConvNet(self.input_size, self.tcn_channels, kernel_size=kernel_size, dropout=dropout)
        self.gru = nn.GRU(self.tcn_channels[-1], self.gru_hidden_size, bidirectional=True, batch_first=True)
        self.informer = Informer(config)
        self.attention = DynamicSparseAttention(self.gru_hidden_size * 2, heads, sparsity)
        self.fc = nn.Linear(self.gru_hidden_size * 2, output_size)

    def forward(self, x):
        hybrid_out = self.hybrid_scale(x)
        hybrid_out = self.informer(hybrid_out)
        tcn_out = self.tcn(hybrid_out.permute(0, 2, 1))
        gru_out, _ = self.gru(tcn_out.permute(0, 2, 1))
        attn_out = self.attention(gru_out, gru_out, gru_out)
        out = self.fc(attn_out[:, -1, :])
        return out
