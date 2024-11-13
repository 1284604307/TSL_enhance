# 解决中文显示问题
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.metrics import metric


# def drawResultCompare(result, real, tag):
#     drawResultCompare(result,real,tag,None)
def drawResultCompare(result, real,tag,savePath):

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制真实值和预测值对比图
    plt.figure(figsize=(12, 8))
    plt.plot(real, label='真实值')
    plt.plot(result, label=f'预测值')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.ylabel('负荷值', fontsize=15)
    plt.xlabel('采样点', fontsize=15)
    plt.title(f"{tag}", fontsize=15)
    plt.show()
    if savePath!=None:
        plt.savefig(f'{savePath}.png')

def saveResultCompare(predicted_values, real,tag):
    # 将两个数组转换为DataFrame，分别作为两列
    data = pd.DataFrame({
        '真实值': real,
        '预测值': predicted_values
    })
    # 将DataFrame保存为csv文件
    data.to_csv(f'../model_result/{tag}.csv', index=False, encoding='utf-8')
    print("CSV 文件已保存")



def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))
def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0)
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def completeMSE(real, predicted):
    # 将列表转换为NumPy数组
    # real = np.reshape(real, -1)
    # predicted = predicted.reshape(-1)
    real = np.array(real)
    prediction = np.array(predicted)
    R2 = r2_score(real, prediction)
    MAE = mean_absolute_error(real, prediction)
    MSE = mean_squared_error(real, prediction)
    RMSE = np.sqrt(MSE)
    MAPE = np.mean(np.abs((real - prediction) / prediction))
    MSPE =  np.mean(np.square((prediction - real) / real))
    # print(f'\n{model_name} 模型评价指标:')
    print(f'R2: {R2:.4f},MSE: {MSE:.4f},MAE: {MAE:.4f}')
    print(f'RMSE: {RMSE:.4f},MAPE: {MAPE:.4f},MSPE: {MSPE:.4f}')
    # print(f',RSE: {RSE(prediction,real):.4f},CORR: {CORR(prediction,real):.4f}')

def metricAndSave(preds, trues,folder_path):
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    np.savetxt(folder_path + 'metrics.txt', np.array([f"mae:{mae}", f"mse:{mse}",f"rmse:{rmse}", f"mape:{mape}", f"mspe:{mspe}"]), fmt='%s')
    np.savetxt(folder_path + 'pred.csv', preds, delimiter=',')
    np.savetxt(folder_path + 'trues.csv', trues, delimiter=',')
    return mae, mse, rmse, mape, mspe