import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.pred_len = configs.pred_len
        # 输入维度为7，隐藏层维度为1，2层LSTM
        # batch_first = True 时 ，入参为`(batch_size, seq_len, input_size)`
        self.lstm = nn.LSTM(input_size=configs.enc_in, hidden_size=32,batch_first=True, num_layers=3, dropout=0.1)
        self.fc = nn.Linear(32, configs.c_out)#可以确定一下对标论文里，是否以下一个时间步数值作为预测目标，对于时序数据来说，可能是预测与上一步的差值，来去噪

    def forward(self, x):
        # x形状为 (batch_size, sequence_length, input_size)
        # LSTM期望输入形状 (sequence_length, batch_size, input_size)
        # x = x.permute(1, 0, 2)
        # 经过LSTM层
        # output, _ = self.lstm(x)
        # # 取最后一个时间步的输出
        # last_output = output[:self.pred_len, :, :]
        # last_output = last_output.permute(1, 2, 0)
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:,:self.pred_len, :])  # 取pred_len时间步的输出
        # out = self.fc(lstm_out[-1,:, :])  # 取最后一个时间步的输出
        return out

