import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def vis_heatmap_with_searbon(weights: np.array):
    '''

    Args:
        weights: [H,W] it must should be 2-dims

    Returns:

    '''
    # sns.heatmap(weights, cmap='hot', annot=True, fmt=".2f")
    sns.heatmap(weights, cmap='hot', annot=False, fmt=".2f")
    plt.show()


def vis_heatmap_with_pure_plt_spatial(matrics: np.array, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                                      cmap="Reds", joint_cnt=17):
    """

    	:param matrics: 输入的矩阵值；例matrics.shape=(1,1,10,10)
    	:param xlabel: 设置x轴的标签
    	:param ylabel: 设置y轴的标签
    	:param titles: 设置表头
    	:param figsize: 设置图片的大小
    	:param cmap: 设置热力图的颜色
    	:return:
    	"""
    # 根据输入矩阵获取行数和列数
    num_rows, num_cols = matrics.shape[0], matrics.shape[1]
    # 设置画布fig和画布区域axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    #######################################################
    # For Joint2joint attention special range(0,17)
    plt.xticks(np.arange(0, joint_cnt, 2))
    plt.yticks(np.arange(0, joint_cnt, 2))
    #######################################################
    # 得到axes，和数据matrix
    for i, (row_axes, row_matrix) in enumerate(zip(axes, matrics)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrix)):
            # 将matrix的值传入到坐标系中
            pcm = ax.imshow(matrix, cmap=cmap)
            # 设置x轴的标签
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            # 设置y轴的标签
            if j == 0:
                ax.set_ylabel(ylabel)
            # 设置表格的抬头
            if titles:
                ax.set_title(titles[j])
    # 给画布添加新的渐变跳
    fig.colorbar(pcm, ax=axes, shrink=0.6)


def vis_heatmap_with_pure_plt_temporal(matrics: np.array, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                                       cmap="Reds", seqlen=243):
    """

    	:param matrics: 输入的矩阵值；例matrics.shape=(1,1,10,10)
    	:param xlabel: 设置x轴的标签
    	:param ylabel: 设置y轴的标签
    	:param titles: 设置表头
    	:param figsize: 设置图片的大小
    	:param cmap: 设置热力图的颜色
    	:return:
    	"""
    # 根据输入矩阵获取行数和列数
    num_rows, num_cols = matrics.shape[0], matrics.shape[1]
    # 设置画布fig和画布区域axes
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                             sharex=True, sharey=True, squeeze=False)

    #######################################################
    # For Joint2joint attention special range(0,T)
    plt.xticks(np.arange(0, seqlen, 50))
    plt.yticks(np.arange(0, seqlen, 50))
    #######################################################
    # 得到axes，和数据matrix
    for i, (row_axes, row_matrix) in enumerate(zip(axes, matrics)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrix)):
            # 将matrix的值传入到坐标系中
            pcm = ax.imshow(matrix, cmap=cmap)
            # 设置x轴的标签
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            # 设置y轴的标签
            if j == 0:
                ax.set_ylabel(ylabel)
            # 设置表格的抬头
            if titles:
                ax.set_title(titles[j])
    # 给画布添加新的渐变跳
    fig.colorbar(pcm, ax=axes, shrink=0.6)


if __name__ == '__main__':
    # 方式一
    # 定义权重矩阵
    weights = np.array([
        [0.2276, 0.2630, 0.2277, 0.2186, 0.0],
        [0.3037, 0.1941, 0.2014, 0.2605, 0.0],
        [0.2428, 0.2346, 0.2105, 0.3160, 0.0],
        [0.2364, 0.2941, 0.2894, 0.1616, 0.0]
    ])
    vis_heatmap_with_searbon(weights)
    # 方式二
    # 设置注意力权重值，torch.eye表示的对角线值为1，其他位置值为0
    attention_weights = np.eye(10).reshape((1, 1, 10, 10))
    vis_heatmap_with_pure_plt_spatial(attention_weights, xlabel="keys", ylabel="queries", titles=['none'])
    plt.show()
