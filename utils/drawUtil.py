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


def getBaseOutputPath(args=None, setting=None):
    if isKaggle():
        path = f"/kaggle/working/"
    else:
        path = f"./results/"
    if (args != None):
        path += args.model + "/"
    if (setting != None):
        path += setting + "/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def saveTxt(path, txt):
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    # 使用 'w' 模式（代表写入文本模式，会覆盖原有内容）打开文件
    with open(path, 'w', encoding='utf-8') as f:
        f.write(txt)


def drawResultCompare(result, real, tag, savePath=None,args=None):
    try:
        if len(real.shape) == 3 and real.shape[2]==1:
            real = real.reshape(real.shape[:-1])
            print("real数组数据长度为3且第三维只有一个元素，缩减数据维度到2")
        if len(result.shape) == 3 and result.shape[2]==1:
            result = result.reshape(result.shape[:-1])
            print("result数组数据长度为3且第三维只有一个元素，缩减数据维度到2")

        # 设置中文字体及解决负号显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        if len(real.shape) == 1:  # 判断数据是否为一维
            # 绘制真实值和预测值对比图（一维情况）
            plt.figure(figsize=(12, 8))
            plt.plot(real, label='真实值')
            plt.plot(result, label='预测值')
            plt.xticks(fontsize=15)
            plt.yticks(fontsize=15)
            plt.legend(loc='best', fontsize=15)
            plt.ylabel('负荷值', fontsize=15)
            plt.xlabel('采样点', fontsize=15)
            plt.title(f"{tag}", fontsize=15)
            plt.show()
            if savePath is not None:
                plt.savefig(f'{savePath}.png')
                print(f"结果对比图保存到{savePath}")
        elif len(real.shape) == 2:  # 判断数据是否为二维
            n_dimensions = real.shape[1]  # 获取维度数量 n_dimensions = pred_len
            # todo 只保存第一个维度结果，避免出图太多
            # for dim in range(n_dimensions):
            for dim in range(1):
                # 为每个维度创建一个新的图（二维情况）
                plt.figure(figsize=(12, 8))
                plt.plot([r[dim] for r in real], label='真实值')
                plt.plot([r[dim] for r in result], label='预测值')
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.legend(loc='best', fontsize=15)
                plt.ylabel(f'负荷值（dim {dim + 1}）', fontsize=15)
                plt.xlabel('采样点', fontsize=15)
                plt.title(f"{tag} - dim {dim + 1}", fontsize=15)
                plt.show()
                if savePath is not None:
                    plt.savefig(f'{savePath}_{dim}.png')
                    print(f"结果对比图保存到{savePath}_{dim}.png")
        else:
            print("数据维度不符合预期，请检查数据格式。")
            print(result)
            print(real)
    except Exception as e:
        print("绘制结果图失败")
        print(e)

def drawResultSample(input_data,pred, real, args):
    count = 0
    for i in range(1,input_data.shape[0],args.pred_len):
        if count>5:
            break
        count+=1
        input_data_sample = input_data[i,:,1]
        pred_sample = pred[i,:]
        real_sample = real[i,:]
        pred_sample = np.concatenate((input_data_sample, pred_sample), axis=0)
        real_sample = np.concatenate((input_data_sample, real_sample), axis=0)

        plt.figure(figsize=(12, 8))
        plt.plot(real_sample, label='真实值')
        plt.plot(pred_sample, label='预测值')
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(loc='best', fontsize=15)
        plt.ylabel('y', fontsize=15)
        plt.xlabel('x', fontsize=15)
        plt.title(f"seq=>pred Sample", fontsize=15)
        plt.show()
        plt.savefig(f'{getBaseOutputPath(args, args.setting)}_Sample案例{i}.png')
        print(f'{getBaseOutputPath(args, args.setting)}_Sample案例{i}.png已保存')

def saveResultCompare(predicted_values, real, basePath):
    # 将两个数组转换为DataFrame，分别作为两列
    np.save(f'{basePath}predicted.npy', predicted_values)
    np.save(f'{basePath}real.npy', real)
    print("npy 文件已保存")
    columns_true = [f'真实值_时间步{i}' for i in range(real.shape[1])]
    columns_pred = [f'预测值_时间步{i}' for i in range(predicted_values.shape[1])]
    columns = columns_true + columns_pred
    data = np.concatenate((real, predicted_values), axis=1)
    df = pd.DataFrame(data, columns=columns)
    # data = pd.DataFrame({
    #     '真实值': real,
    #     '预测值': predicted_values
    # })
    # 将DataFrame保存为csv文件
    df.to_csv(f'{basePath}预测结果.csv', index=False, encoding='utf-8')
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
    try:
        if len(real.shape) == 3 and real.shape[2]==1:
            real = real.reshape(real.shape[:-1])
            print("real数组数据长度为3且第三维只有一个元素，缩减数据维度到2")
        if len(predicted.shape) == 3 and predicted.shape[2]==1:
            predicted = predicted.reshape(predicted.shape[:-1])
            print("predicted数组数据长度为3且第三维只有一个元素，缩减数据维度到2")
        print("\033[1m" + "Complete 预测效果" + "\033[0m")
        print(f'  {"标签形状:":<10}{str(real.shape):<20}  {"预测结果形状:":<10}{str(predicted.shape):<20}')
        real = np.array(real)
        prediction = np.array(predicted)
        R2 = r2_score(real, prediction)
        MAE = mean_absolute_error(real, prediction)
        MSE = mean_squared_error(real, prediction)
        RMSE = np.sqrt(MSE)
        MAPE = np.mean(np.abs((real - prediction) / prediction))
        MSPE = np.mean(np.square((prediction - real) / real))
        # print(f'\n{model_name} 模型评价指标:')
        resultStr = ""
        resultStr += f'  {"R2:":<10}{R2:<20}  {"MSE:":<10}{MSE:<20}\n'
        resultStr += f'  {"MAE:":<10}{MAE:<20}  {"RMSE:":<10}{RMSE:<20}\n'
        resultStr += f'  {"MAPE:":<10}{MAPE:<20}  {"MSPE:":<10}{MSPE:<20}\n'
        print(resultStr)
        return resultStr
    except Exception as e:
        print("绘制结果图失败")
        print(e)
    # print(f'  {"R2:":<10}{R2:<20}  {"MSE:":<10}{MSE:<20}')
    # print(f'  {"MAE:":<10}{MAE:<20}  {"RMSE:":<10}{RMSE:<20}')
    # print(f'  {"MAPE:":<10}{MAPE:<20}  {"MSPE:":<10}{MSPE:<20}')
    # print(f',RSE: {RSE(prediction,real):.4f},CORR: {CORR(prediction,real):.4f}')


def metricAndSave(preds, trues, folder_path):
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    resultStr = completeMSE(preds, trues)
    saveTxt(folder_path + 'metrics.txt', resultStr)
    np.savetxt(folder_path + 'pred.csv', preds, delimiter=',')
    np.savetxt(folder_path + 'trues.csv', trues, delimiter=',')
    print("结果已保存到:{}".format(folder_path + 'metrics.txt'))
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
