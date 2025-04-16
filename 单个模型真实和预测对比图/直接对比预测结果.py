import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils import drawUtil

# 读取文件中的真实值和预测值
real = []  # 存储真实值的一维数组
prediction = []  # 存储预测值的一维数组

path = '../results/'
settings = 'BNN/bnn_[20221201_20241201_Federal_utc]_96_1_BNN_conv_ETTh1_ftMS_sl96_ll1_pl1_dm16_nh8_el2_dl1_df2048_expand2_dc4_fc6_ebtimeF_dtTrue_Exp_0'

pred_path = path + settings +'/pred.csv'
true_path = path + settings +'/trues.csv'

preds = pd.read_csv(pred_path)
# 取第一个
preds = preds.values[:,0]
trues = pd.read_csv(true_path)

real = trues.values.reshape(-1)
prediction = preds.reshape(-1)
# 打印读取的结果
# print("True values:", real)
# print("Predicted values:", prediction)

# 将列表转换为NumPy数组
# real = np.array(real)
# prediction = np.array(prediction)

R2 = r2_score(real, prediction)
MAE = mean_absolute_error(real, prediction)
RMSE = np.sqrt(mean_squared_error(real, prediction))

# print(f'\n{model_name} 模型评价指标:')
print(f'R2: {R2:.4f}')
print(f'MAE: {MAE:.4f}')
print(f'RMSE: {RMSE:.4f}')
MAPE = np.mean(np.abs((real - prediction) / prediction))
print(f'MAPE: {MAPE:.4f}')

drawUtil.drawResultCompare(result=prediction,real=real,tag="prediction",savePath=None)
drawUtil.completeMSE(prediction,real)