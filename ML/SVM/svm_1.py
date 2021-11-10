# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/5 11:12 
# License: bupt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")  # rainbow彩虹色

# 1. 绘制散点图，首先要有散点图
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")
ax = plt.gca()  # 获取当前的子图，如果不存在，则创建新的子图
# plt.show()

# 2. 绘制形成带有背景的网状图
# 获取平面上两条坐标轴的最大值和最小值
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 在最大值和最小值之间形成30个规律的数据
axisx = np.linspace(xlim[0], xlim[1], 30)
axisy = np.linspace(ylim[0], ylim[1], 30)

axisy, axisx = np.meshgrid(axisy, axisx)
# 我们将使用这里形成的二维数组作为我们contour函数中的X和Y
# 使用meshgrid函数将两个一维向量转换为特征矩阵
# 核心是将两个特征向量广播，以便获取y.shape * x.shape这么多个坐标点的横坐标和纵坐标

xy = np.vstack([axisx.ravel(), axisy.ravel()]).T
# 其中ravel()是降维函数，vstack能够将多个结构一致的一维数组按行堆叠起来
# xy就是已经形成的网格，它是遍布在整个画布上的密集的点

plt.scatter(xy[:, 0], xy[:, 1], s=1, cmap="rainbow")
# plt.show()

# 3. 建模，通过fit计算出对应的决策边界
clf = SVC(kernel="linear").fit(X, y)  # 计算出对应的决策边界
Z = clf.decision_function(xy).reshape(axisx.shape)
# 重要接口decision_function，返回每个输入的样本所对应的到决策边界的距离
# 然后再将这个距离转换为axisx的结构，这是由于画图的函数contour要求Z的结构必须与X和Y保持一致

# 首先要有散点图
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")
ax = plt.gca()  # 获取当前的子图，如果不存在，则创建新的子图
# 画决策边界和平行于决策边界的超平面
ax.contour(axisx, axisy, Z
           , colors="k"
           , levels=[-1, 0, 1]  # 画三条等高线，分别是Z为-1，Z为0和Z为1的三条线
           , alpha=0.5  # 透明度
           , linestyles=["--", "-", "--"])

# 设置x轴取值
ax.set_xlim(xlim)
ax.set_ylim(ylim)
plt.show()

# 3.1 svc常用方法
clf.predict(X)
# 根据决策边界，对X中的样本进行分类，返回的结构为n_samples

clf.score(X, y)
# 返回给定测试数据和标签的平均准确度

clf.support_vectors_
# 返回支持向量坐标

clf.n_support_  # array([2, 1])


# 返回每个类中支持向量的个数

# 3.2 将上述过程包装成函数：
def plot_svc_decision_function(model, ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()


# 3.3 导入一部分为圆的数据
from sklearn.datasets import make_circles

X, y = make_circles(100, factor=0.1, noise=.1)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")
plt.show()
# 3.5 利用封装好的距离进行测试
clf = SVC(kernel="linear").fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")
# plot_svc_decision_function(clf)
print(clf.score(X, y))

# 定义一个由x计算出来的新维度r
r = np.exp(-(X ** 2).sum(1))

rlim = np.linspace(min(r), max(r), 100)

from mpl_toolkits import mplot3d


# 3.6 定义一个绘制三维图像的函数
# elev表示上下旋转的角度
# azim表示平行旋转的角度
def plot_3D(elev=30, azim=30, X=X, y=y):
    ax = plt.subplot(projection="3d")
    ax.scatter3D(X[:, 0], X[:, 1], r, c=y, s=50, cmap='rainbow')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("r")
    plt.show()


# 3.7  对于圆形的数据进行处理plot_3D()
clf = SVC(kernel="rbf").fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="rainbow")
plot_svc_decision_function(clf)
print(clf.score(X, y))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm  # from sklearn.svm import SVC  两者都可以
from sklearn.datasets import make_circles, make_moons, make_blobs, make_classification

# 4. 导入4部分数据，分别查看效果
n_samples = 100

datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=0),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=1),
    make_blobs(n_samples=n_samples, centers=2, random_state=5),  # 分簇的数据集
    make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, random_state=5)
    # n_features：特征数，n_informative：带信息的特征数，n_redundant：不带信息的特征数
]

Kernel = ["linear", "poly", "rbf", "sigmoid"]

# 4.1 四个数据集分别是什么样子呢？
for X, Y in datasets:
    plt.figure(figsize=(5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap="rainbow")
    plt.show()

nrows = len(datasets)
ncols = len(Kernel) + 1

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))
plt.show()

nrows = len(datasets)
ncols = len(Kernel) + 1

fig, axes = plt.subplots(nrows, ncols, figsize=(20, 16))

# 第一层循环：在不同的数据集中循环
for ds_cnt, (X, Y) in enumerate(datasets):

    # 在图像中的第一列，放置原数据的分布
    ax = axes[ds_cnt, 0]
    if ds_cnt == 0:
        ax.set_title("Input data")
    ax.scatter(X[:, 0], X[:, 1], c=Y, zorder=10, cmap=plt.cm.Paired, edgecolors='k')
    ax.set_xticks(())
    ax.set_yticks(())

    # 第二层循环：在不同的核函数中循环
    # 从图像的第二列开始，一个个填充分类结果
    for est_idx, kernel in enumerate(Kernel):

        # 定义子图位置
        ax = axes[ds_cnt, est_idx + 1]

        # 建模
        clf = svm.SVC(kernel=kernel, gamma=2).fit(X, Y)
        score = clf.score(X, Y)

        # 绘制图像本身分布的散点图
        ax.scatter(X[:, 0], X[:, 1], c=Y
                   , zorder=10
                   , cmap=plt.cm.Paired, edgecolors='k')
        # 绘制支持向量
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=50,
                   facecolors='none', zorder=10, edgecolors='k')  # facecolors='none':透明的

        # 绘制决策边界
        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

        # np.mgrid，合并了我们之前使用的np.linspace和np.meshgrid的用法
        # 一次性使用最大值和最小值来生成网格
        # 表示为[起始值：结束值：步长]
        # 如果步长是复数，则其整数部分就是起始值和结束值之间创建的点的数量，并且结束值被包含在内
        XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
        # np.c_，类似于np.vstack的功能
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()]).reshape(XX.shape)
        # 填充等高线不同区域的颜色
        ax.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
        # 绘制等高线
        ax.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                   levels=[-1, 0, 1])

        # 设定坐标轴为不显示
        ax.set_xticks(())
        ax.set_yticks(())

        # 将标题放在第一行的顶上
        if ds_cnt == 0:
            ax.set_title(kernel)

        # 为每张图添加分类的分数
        ax.text(0.95, 0.06, ('%.2f' % score).lstrip('0')
                , size=15
                , bbox=dict(boxstyle='round', alpha=0.8, facecolor='white')
                # 为分数添加一个白色的格子作为底色
                , transform=ax.transAxes  # 确定文字所对应的坐标轴，就是ax子图的坐标轴本身
                , horizontalalignment='right'  # 位于坐标轴的什么方向
                )

plt.tight_layout()
plt.show()




