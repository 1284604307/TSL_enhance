import torch as th

from kan import KAN
from torch import nn


class Model(th.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.inputdim = args.enc_in
        self.outdim = args.c_out
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        # self.model = KAN(width=[self.inputdim, 16], grid=3, k=3, seed=1)
        # self.linear = nn.Linear(self.inputdim * self.seq_len, self.inputdim)
        # self.model = KAN(width=[self.inputdim, args.pred_len * args.c_out ], grid=3, k=3, seed=1)
        self.model = KAN(width=[self.inputdim* self.seq_len, args.pred_len * args.c_out ], grid=3, k=3, seed=1)

    def forward(self, x):
        x = x.view(x.shape[0],-1)
        # x = self.linear(x)
        return self.model.forward(x)

