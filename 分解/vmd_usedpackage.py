import pandas as pd
import numpy as np
from vmdpy import VMD
import matplotlib.pyplot as plt

# 读取CSV数据
# root_path = 'D:\深度学习\代码\A数据集/'  # 替换为你的文件路径
root_path = 'D:\深度学习\代码\A数据集\AEMO澳大利亚能源市场运营商电力需求数据集/'
file_path = "AEMO_QLD_electricity_processed.csv"
# file_path = "ETT-small/ETTh1.csv"
file_path = root_path + file_path
dataset_name = file_path.split("/").pop().replace(".csv", "")

data = pd.read_csv(file_path)

# 假设CSV文件有两个列：'时间' 和 '电力'
date = data['SETTLEMENTDATE']  # 时间列
time = data['Time']  # 时间列
target = data['TOTALDEMAND']  # 目标列

# 将电力数据转换为numpy数组，用于VMD分解
target_signal = target.values

# VMD 参数设置
alpha = 2000  # 带宽限制（IMF的带宽）
tau = 0  # 噪声容忍程度（一般取0）
K = 12  # 模态数量）
DC = 0  # 强制DC模态（第一个分量是不是一个直流分量）
init = 1  # 初始化（每个IMF的中心频率初始值  0：都为0  1：均匀分布  2：随机的）
tol = 1e-7  # 收敛精度/容差

# 执行VMD分解
# u=>得到的IMFs
# u_hat=>IMFs频谱
# omega=>估计的IMFs中心频率
u, u_hat, omega = VMD(target_signal, alpha, tau, K, DC, init, tol)

# 打印VMD分解后的模态
for i in range(K):
    plt.figure(figsize=(10, 4))
    plt.plot(time, u[i], label=f'IMF {i + 1}')
    plt.title(f'Intrinsic Mode Function {i + 1}')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# 如果需要保存每个模态数据
for i in range(K):
    data[f'IMF_{i + 1}'] = u[i]

# 保存新文件包含VMD结果
data.to_csv(f'vmd_output_{dataset_name}.csv', index=False)
