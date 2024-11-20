import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import drawUtil


# 使用线性插值预处理数据，弥补缺失值及异常值
def processMissDataFromPath(path):
    #读取原始数据
    # "光伏数据.xlsx"
    raw_data = pd.read_excel(path)
    return processMissDataFromRawData(raw_data)


def processMissDataFromRawData(raw_data):
    # 显示数据集的基本信息
    data_info = raw_data.info()
    print(data_info)
    selected_data = raw_data
    selected_data.reset_index(drop=True, inplace=True)
    selected_data.head()
    # 选择数据集中的数值列
    numeric_cols = selected_data.select_dtypes(include=['number'])
    # 对这些数值列进行线性插值
    numeric_cols_interpolated = numeric_cols.interpolate(method='linear')
    # 将插值后的数值列重新合并到原始数据集中
    selected_data_interpolated = selected_data.copy()
    selected_data_interpolated[numeric_cols.columns] = numeric_cols_interpolated
    #selected_data_interpolated.head()
    selected_data_interpolated.info()
    return selected_data_interpolated

def processErrorDataFromRawData(data):

    drawUtil.drawBBox(data,figPath="2_异常值检测_箱型图_前.png")

    # 定义异常值的阈值（可以根据具体情况调整）
    threshold = 1.5

    # 遍历每一列，检测异常值并替换为缺失值
    for column in data.columns:
        # 计算箱型图的四分位距
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1

        # 计算异常值的上下界
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # 将超出上下界的值设为缺失值
        data.loc[(data[column] < lower_bound) |
                                       (data[column] > upper_bound), column] = np.nan
    print(data.info())

    # 对所有数值列进行线性插值填充缺失值
    selected_data_interpolated_linear = data.interpolate(method='linear')
    print(selected_data_interpolated_linear.info())

    drawUtil.drawBBox(data,figPath="2_异常值检测_箱型图_后.png")

    return selected_data_interpolated_linear
