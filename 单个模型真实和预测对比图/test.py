import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体及解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 模拟数据，假设你有100个标本
num_samples = 100


real = np.sin(np.linspace(0, 10, num_samples)) + 0.2 * np.random.randn(num_samples)
pred_mean = np.sin(np.linspace(0, 10, num_samples))
pred_var = 0.1 * np.ones(num_samples)
pred = np.column_stack((pred_mean, pred_var))

# 创建画布和子图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# 计算置信区间
lower_bound = pred[:, 0] - 2 * np.sqrt(pred[:, 1])
upper_bound = pred[:, 0] + 2 * np.sqrt(pred[:, 1])

# 第一个子图（上方）
ax1.plot(np.arange(num_samples), real, c='k', linewidth=1, alpha=0.5, label='真实值')
ax1.plot(np.arange(num_samples), pred[:, 0], c='r', linewidth=1, label='预测均值')
ax1.fill_between(np.arange(num_samples), lower_bound, upper_bound, color='skyblue', alpha=0.3, label='置信区间')
ax1.set_xlabel('标本索引')
ax1.set_ylabel('值')
ax1.set_title('真实值与预测值对比')
ax1.legend()

# 第二个子图（下方）
ax2.plot(np.arange(num_samples), real, c='k', linewidth=1, alpha=0.5, label='真实值')
ax2.plot(np.arange(num_samples), pred[:, 0], c='r', linewidth=1, label='预测均值')
ax2.fill_between(np.arange(num_samples), lower_bound, upper_bound, color='skyblue', alpha=0.3, label='置信区间')
ax2.set_xlabel('标本索引')
ax2.set_ylabel('值')
ax2.set_title('另一种视角的对比')
ax2.legend()

# 调整布局并显示图形
plt.tight_layout()
plt.show()


def drawResultCompareWithMeanAndVariance(result, real, tag, savePath=None,args=None):
    pred = result
    try:
        # 设置中文字体及解决负号显示问题
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        pred = pred[:1000]  # 确保方差为正
        real = real[:1000]  # 确保方差为正
        num_samples = len(real)


        # real数据维度 【标本数量，实际值】
        # result数据维度 【标本数量，【均值方差】】
        if(len(real.shape) == 2 and len(result.shape) == 2):
            # pred[:, 1] = np.abs(pred[:, 1])  # 确保方差为正

            # 创建画布和子图
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

            # 计算置信区间
            lower_bound = pred[:, 0] - 2 * np.sqrt(pred[:, 1])
            upper_bound = pred[:, 0] + 2 * np.sqrt(pred[:, 1])

            # 第一个子图（上方）
            ax1.plot(np.arange(num_samples), real, c='k', linewidth=1, alpha=0.5, label='真实值')
            ax1.plot(np.arange(num_samples), pred[:, 0], c='r', linewidth=1, label='预测均值')
            ax1.fill_between(np.arange(num_samples), lower_bound, upper_bound, color='skyblue', alpha=0.3,
                             label='置信区间')
            ax1.set_xlabel('标本索引')
            ax1.set_ylabel('值')
            ax1.set_title('真实值与预测值对比')
            ax1.legend()

            # 第二个子图（下方）
            ax2.plot(np.arange(num_samples), real, c='k', linewidth=1, alpha=0.5, label='真实值')
            ax2.plot(np.arange(num_samples), pred[:, 0], c='r', linewidth=1, label='预测均值')
            ax2.fill_between(np.arange(num_samples), lower_bound, upper_bound, color='skyblue', alpha=0.3,
                             label='置信区间')
            ax2.set_xlabel('标本索引')
            ax2.set_ylabel('值')
            ax2.set_title('另一种视角的对比')
            ax2.legend()

            # 调整布局并显示图形
            plt.tight_layout()
            plt.show()


            if savePath is not None:
                plt.savefig(f'{savePath}.png')
                print(f"结果对比图保存到{savePath}")
        else:
            print("数据维度不符合预期，请检查数据格式。")
            print(result)
            print(real)
    except Exception as e:
        print("绘制结果图失败")
        print(e)

