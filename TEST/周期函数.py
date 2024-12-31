from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x = np.linspace(0, 120, 6000)
y = np.sin((np.pi/6)*x+3)
print(x)
print(y)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('y = sin(π/3*x) with period 6')# 创建DataFrame
start_date = datetime(2024, 1, 1)
date_list = [start_date + timedelta(days=i) for i in range(len(x))]
df = pd.DataFrame({'x': x, 'y': y, 'date': date_list})
# df = pd.DataFrame({'x': x, 'y': y})
# 保存到Excel文件
plt.show()
df.to_csv('D:\深度学习\代码\A数据集\电力数据集\sinx.csv', index=False)
