import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# class TemporalConvNet(nn.Module):
class MLP(nn.Module):
    def __init__(self, num_inputs,c_out=1,seq_len=96 , pred_len=1):
        super(MLP, self).__init__()
        self.pred_len = pred_len
        self.c_out = c_out
        self.transferLayer = nn.modules.Linear(in_features= num_inputs*seq_len , out_features=pred_len*c_out)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.transferLayer(x)
        out = out.view(out.size(0), self.pred_len, self.c_out)
        return out


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.model = MLP(configs.enc_in, c_out=configs.c_out,seq_len=configs.seq_len,pred_len=configs.pred_len)

    def forward(self, x):
        return self.model.forward(x)
