from torch import nn


# 混合尺度特征提取
class HybridScaleFeatureExtraction(nn.Module):
    def __init__(self, embed_size):
        super(HybridScaleFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_size, out_channels=embed_size, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels=embed_size, out_channels=embed_size, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=embed_size, out_channels=embed_size, kernel_size=5, padding=2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        out = conv1_out + conv3_out + conv5_out
        out = out.permute(0, 2, 1)
        return out
