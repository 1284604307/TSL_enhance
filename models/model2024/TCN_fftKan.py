import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from models.fftKAN import Model as KAN


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
        Temporal Convolutional Networks
        num_inputs: 输入的特征数量
        num_channels: num_channels是一个数组，记录了每层的hidden_channel数，其长度len(num_channels)决定了TemporalBlock深度（数量）
        kernel_size: 卷积核尺寸
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# class TemporalConvNet(nn.Module):
class TemporalConvNet_effKAN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, c_out=1,seq_len=96 , pred_len=1):
        super(TemporalConvNet_effKAN, self).__init__()
        layers = []
        self.pred_len = pred_len
        # 增加一个转换Block，使输出的特征数 = 需要的特征数
        # num_channels.append(c_out)
        # todo
        # num_channels.append(pred_len)
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        # 增加一个转换层，使输出的时间步 = 需要的时间步
        # 增加一个转换层，使输出的结果 = 需要的结果长度( 1 或 特征长度)
        self.transferLayer = nn.modules.Linear(in_features= seq_len , out_features=c_out)
        # todo 配置KAN隐层数量  最后应该输出多个特征还是多个时间步【有待考虑】
        self.e_kan = KAN(num_channels[-1],pred_len,gridsize=3)

    def forward(self, x):
        # 置换2，3，todo 为了按单时间步特征通道卷积训练
        x = x.permute(0, 2, 1)

        out = self.network(x)
        out = self.transferLayer(out)
        out = self.e_kan(out[:,:,-1])

        # if len(batch_y.shape)<len(outputs.shape):
        #     # 补充一层，使模型输出shape = y shape
        #     batch_y = batch_y.unsqueeze(-1)
        # # todo 多对多预测，调换预测的通道次序
        # if(self.args.features.endswith("M")):
        #     batch_y = batch_y.permute(0, 2, 1)
        return out


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.model = TemporalConvNet_effKAN(configs.enc_in, configs.num_channels, configs.kernel_size, configs.dropout
                                     , c_out=configs.c_out,seq_len=configs.seq_len,pred_len=configs.pred_len)
    def forward(self, x):
        return self.model.forward(x)
