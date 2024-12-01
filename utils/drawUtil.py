# 解决中文显示问题
import os

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from utils.metrics import metric
import seaborn as sns


def isKaggle():
    import os
    # 检查是否在 Kaggle 环境中
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return True
        # if os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive':
        #     print("在 Kaggle Notebook 中运行")
        # elif os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Batch':
        #     print("在 Kaggle Batch Environment 中运行")
    else:
        return False


def getBaseOutputPath():
    if isKaggle():
        return "/kaggle/working/"
    else:
        return "./"


def saveTxt(path, txt):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 使用 'w' 模式（代表写入文本模式，会覆盖原有内容）打开文件
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)


def drawResultCompare(result, real, tag, savePath=None):
    try:
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
        if savePath != None:
            plt.savefig(f'{savePath}.png')
            print(f"结果对比图保存到{savePath}")
    except Exception as e:
        print("绘制结果图失败")
        print(e)


def saveResultCompare(predicted_values, real, tag):
    # 将两个数组转换为DataFrame，分别作为两列
    data = pd.DataFrame({
        '真实值': real,
        '预测值': predicted_values
    })
    # 将DataFrame保存为csv文件
    data.to_csv(f'{getBaseOutputPath()}model_result/{tag}.csv', index=False, encoding='utf-8')
    print("CSV 文件已保存")


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


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
    MSPE = np.mean(np.square((prediction - real) / real))
    # print(f'\n{model_name} 模型评价指标:')
    print(f'R2: {R2:.4f},MSE: {MSE:.4f},MAE: {MAE:.4f}')
    print(f'RMSE: {RMSE:.4f},MAPE: {MAPE:.4f},MSPE: {MSPE:.4f}')
    # print(f',RSE: {RSE(prediction,real):.4f},CORR: {CORR(prediction,real):.4f}')


def metricAndSave(preds, trues, folder_path):
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print('mse:{}, mae:{}'.format(mse, mae))
    np.savetxt(folder_path + 'metrics.txt',
               np.array([f"mae:{mae}", f"mse:{mse}", f"rmse:{rmse}", f"mape:{mape}", f"mspe:{mspe}"]), fmt='%s')
    np.savetxt(folder_path + 'pred.csv', preds, delimiter=',')
    np.savetxt(folder_path + 'trues.csv', trues, delimiter=',')
    print()
    return mae, mse, rmse, mape, mspe


def drawBBox(rawData, figPath="箱型图.png"):
    # 绘制箱型图
    rc = {'font.sans-serif': 'SimHei',
          'axes.unicode_minus': False}
    # 设置Seaborn的风格
    sns.set_style("whitegrid", rc=rc)

    plt.figure(figsize=(15, 10))

    # 绘制每个数值列的箱型图
    for i, column in enumerate(rawData.columns, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(y=rawData[column], color='purple')
        plt.title(column)

    plt.tight_layout()
    plt.savefig(getBaseOutputPath() + figPath, dpi=600, bbox_inches='tight')
    plt.show()
