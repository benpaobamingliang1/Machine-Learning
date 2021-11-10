# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/8 10:30 
# License: bupt

# 1. 导库
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# 2. 加载数据
iris = load_iris()
y = iris.target
X = iris.data
# 作为数组，X是几维？
X.shape  # (150, 4)
# 作为数据表或特征矩阵，X是几维？
import pandas as pd

pd.DataFrame(X).head()

# 3. 建立模型
# 也可以fit_transform一步到位
# 调用PCA
pca = PCA(n_components=2)  # 实例化
pca = pca.fit(X)  # 拟合模型
X_dr = pca.transform(X)  # 获取新矩阵
# X_dr = PCA(2).fit_transform(X)

# 4. 要将三种鸢尾花的数据分布显示在二维平面坐标系中，对应的两个坐标（两个特征向量）应该是三种鸢尾花降维后的x1和x2，怎样才能取出三种鸢尾花下不同的x1和x2呢？
# 对于每一列中的数据进行判断，如果为0就写入到数据之中
# print(X_dr[y == 0, 0])

# 要展示三中分类的分布，需要对三种鸢尾花分别绘图
# 可以写成三行代码，也可以写成for循环
"""
plt.figure()
plt.scatter(X_dr[y==0, 0], X_dr[y==0, 1], c="red", label=iris.target_names[0])
plt.scatter(X_dr[y==1, 0], X_dr[y==1, 1], c="black", label=iris.target_names[1])
plt.scatter(X_dr[y==2, 0], X_dr[y==2, 1], c="orange", label=iris.target_names[2])
plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()
"""

colors = ['red', 'black', 'orange']
iris.target_names

plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_dr[y == i, 0]
                , X_dr[y == i, 1]
                , alpha=.7  # 指画出的图像的透明度
                , c=colors[i]
                , label=iris.target_names[i]
                )
plt.legend()  # 图例
plt.title('PCA of IRIS dataset')
plt.show()

# 属性explained_variance_，查看降维后每个新特征向量上所带的信息量大小（可解释性方差的大小）
pca.explained_variance_  # 查看方差是否从大到小排列，第一个最大，依次减小   array([4.22824171, 0.24267075])

# 属性explained_variance_ratio，查看降维后每个新特征向量所占的信息量占原始数据总信息量的百分比
# 又叫做可解释方差贡献率
pca.explained_variance_ratio_  # array([0.92461872, 0.05306648])
# 大部分信息都被有效地集中在了第一个特征上

pca.explained_variance_ratio_.sum()  # 0.977685206318795

import numpy as np

pca_line = PCA().fit(X)
# pca_line.explained_variance_ratio_#array([0.92461872, 0.05306648, 0.01710261, 0.00521218])
plt.plot([1, 2, 3, 4], np.cumsum(pca_line.explained_variance_ratio_))
plt.xticks([1, 2, 3, 4])  # 这是为了限制坐标轴显示为整数
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()

pca_mle = PCA(n_components="mle")  # mle缺点计算量大
pca_mle = pca_mle.fit(X)
X_mle = pca_mle.transform(X)

X_mle  # 3列的数组
# 可以发现，mle为我们自动选择了3个特征

pca_mle.explained_variance_ratio_.sum()  # 0.9947878161267247
# 得到了比设定2个特征时更高的信息含量，对于鸢尾花这个很小的数据集来说，3个特征对应这么高的信息含量，并不
# 需要去纠结于只保留2个特征，毕竟三个特征也可以可视化

pca_f = PCA(n_components=0.97, svd_solver="full")  # svd_solver="full"不能省略
pca_f = pca_f.fit(X)
X_f = pca_f.transform(X)
X_f
pca_f.explained_variance_ratio_  # array([0.92461872, 0.05306648])

# X.shape()#(m,n)
PCA(2).fit(X).components_.shape  # (2, 4)

PCA(2).fit(X).components_  # V(k,n)
# array([[ 0.36138659, -0.08452251,  0.85667061,  0.3582892 ],
#        [ 0.65658877,  0.73016143, -0.17337266, -0.07548102]])


from sklearn.datasets import fetch_lfw_people  # 7个人的1000多张人脸图片组成的一组人脸数据
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# 实例化   min_faces_per_person=60：每个人取出60张脸图
faces = fetch_lfw_people(min_faces_per_person=60)
# 字典形式的数据

# （1277,62,47）  1277是矩阵中图像的个数，62是每个图像的特征矩阵的行，47是每个图像的特征矩阵的列
# 怎样理解这个数据的维度？
faces.images.shape
# （1277,2914）   行是样本，列是样本相关的所有特征：2914 = 62 * 47
# 换成特征矩阵之后，这个矩阵是什么样？
faces.data.shape
X = faces.data

# 数据本身是图像，和数据本身只是数字，使用的可视化方法不同

# 创建画布和子图对象
fig, axes = plt.subplots(4, 5  # 4行5列个图
                         , figsize=(8, 4)  # figsize指的是图的尺寸
                         , subplot_kw={"xticks": [], "yticks": []}  # 不要显示坐标轴
                         )
fig  # 指的是画布

axes
# 不难发现，axes中的一个对象对应fig中的一个空格
# 我们希望，在每一个子图对象中填充图像（共24张图），因此我们需要写一个在子图对象中遍历的循环
axes.shape  # （4,5）

# 二维结构，可以有两种循环方式，一种是使用索引，循环一次同时生成一列上的四个图
# 另一种是把数据拉成一维，循环一次只生成一个图
# 在这里，究竟使用哪一种循环方式，是要看我们要画的图的信息，储存在一个怎样的结构里
# 我们使用 子图对象.imshow 来将图像填充到空白画布上
# 而imshow要求的数据格式必须是一个(m,n)格式的矩阵，即每个数据都是一张单独的图
# 因此我们需要遍历的是faces.images，其结构是(1277, 62, 47)
# 要从一个数据集中取出24个图，明显是一次性的循环切片[i,:,:]来得便利
# 因此我们要把axes的结构拉成一维来循环

# [*axes.flat]#2维
axes.flat  # 降低一个维度
# [*axes.flat] #1维


# 填充图像
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i, :, :]
              , cmap="summer"  # 选择色彩的模式
              )

plt.show()
# cmap参数取值选择各种颜色：https://matplotlib.org/tutorials/colors/colormaps.html

# 原本有2900维，我们现在来降到150维
pca = PCA(150).fit(X)  # 这里X = faces.data，不是faces.images.shape ,因为sklearn只接受2维数组降，不接受高维数组降
# x_dr = pca.transform(X)
# x_dr.shape#(1277,150)

V = pca.components_  # 新特征空间
V.shape  # V（k，n）   (150, 2914)

# 原本有2900维，我们现在来降到150维
pca = PCA(150).fit(X)  # 这里X = faces.data，不是faces.images.shape ,因为sklearn只接受2维数组降，不接受高维数组降
# x_dr = pca.transform(X)
# x_dr.shape#(1277,150)

# 新特征空间
V = pca.components_
# V（k，n）   (150, 2914)
V.shape

fig, axes = plt.subplots(3, 8, figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
    ax.imshow(V[i, :].reshape(62, 47), cmap="gray")

plt.show()

# fig, axes = plt.subplots(3,8,figsize=(8,4),subplot_kw = {"xticks":[],"yticks":[]})
#
# for i, ax in enumerate(axes.flat):
#     ax.imshow(V[i,:].reshape(62,47),cmap="gray")


#  '''继续提升此模型拟合效果'''
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

faces = fetch_lfw_people(min_faces_per_person=60)
faces.images.shape
# 怎样理解这个数据的维度？
faces.data.shape
# 换成特征矩阵之后，这个矩阵是什么样？
X = faces.data

pca = PCA(150)  # 实例化
X_dr = pca.fit_transform(X)  # 拟合+提取结果

print(X_dr.shape)

X_inverse = pca.inverse_transform(X_dr)

X_inverse.shape  # (1348, 2914)

faces.images.shape  # (1348, 62, 47)

faces.images.shape

fig, ax = plt.subplots(2, 10, figsize=(10, 2.5)
                       , subplot_kw={"xticks": [], "yticks": []}
                       )

# 和2.3.3节中的案例一样，我们需要对子图对象进行遍历的循环，来将图像填入子图中
# 那在这里，我们使用怎样的循环？
# 现在我们的ax中是2行10列，第一行是原数据，第二行是inverse_transform后返回的数据
# 所以我们需要同时循环两份数据，即一次循环画一列上的两张图，而不是把ax拉平

for i in range(10):
    ax[0, i].imshow(faces.images[i, :, :], cmap="binary_r")
    ax[1, i].imshow(X_inverse[i].reshape(62, 47), cmap="binary_r")

plt.show()

# 3. 手写数字识别
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
digits.data.shape  # (1797, 64)
# 查看target有哪几个数  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
print(set(digits.target.tolist()))

digits.images.shape  # (1797, 8, 8)


def plot_digits(data):
    # data的结构必须是（m,n），并且n要能够被分成（8,8）这样的结构
    fig, axes = plt.subplots(4, 10, figsize=(10, 4)
                             , subplot_kw={"xticks": [], "yticks": []}
                             )
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8), cmap="binary")
    plt.show()


# plot_digits(digits.data)

rng = np.random.RandomState(42)

# 在指定的数据集中，随机抽取服从正态分布的数据
# 两个参数，分别是指定的数据集，和抽取出来的正太分布的方差
noisy = rng.normal(digits.data, 2)  # np.random.normal(digits.data,2)

plot_digits(noisy)

pca = PCA(0.5, svd_solver='full').fit(noisy)
X_dr = pca.transform(noisy)
X_dr.shape  # (1797, 6)

without_noise = pca.inverse_transform(X_dr)
plot_digits(without_noise)

