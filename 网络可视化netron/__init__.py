import netron
import torch

from models.model2024.TCN_fftKan import  Model as TCN_fftKan
# 实例化网络和输入形状
model = TCN_fftKan({})
input_sample = torch.randn(1, 3, 64, 64)

# 方式一 保存pth文件 并让netron读取
# model_path = 'cross.pth'
# torch.save(model.state_dict(),model_path)

# 方式二  导出onnx格式并读取
torch.onnx.export(model,input_sample,f='cross.onnx')
netron.start('cross.onnx')