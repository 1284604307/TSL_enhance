import matplotlib.pyplot as plt
import numpy as np

# 生成横坐标数据，这里生成0到10之间均匀分布的100个点
x = np.linspace(0, 10, 100)

# 生成随机噪声，使用正态分布来模拟随机噪声，均值为0，标准差为1，形状和x一致
noise = np.random.normal(1, 5, size=x.shape)

# 生成纵坐标数据，这里简单地以y = x + 噪声的形式来构造带有随机变化的数据
y = x + noise

# 绘制曲线
plt.plot(x, y)

# 添加图表标题、坐标轴标签
plt.title("Random Curve")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# 显示图表
plt.show()