# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/9/11 13:31 
# License: bupt
import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 算法实现,定义距离实现函数,引入欧式距离
from scipy.spatial.distance import cdist

class K_Mean(object):
    # 初始化,参数n_cluster(K), 迭代次数max_iter, 初始质心centroids
    def __init__(self, n_cluster=2, max_iter=300, centroids=[]):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.centroids = np.array(centroids, dtype=np.float64)
    #训练模型方法，k-mean聚类过程 ,传入原始数据
    def fit(self, data):
        # 假如没有指定初始质心，就随机选取data中的点作为初始质心
        if (self.centroids.shape == (0,)):
            # 从data中随机生成0到data行数的6个整数，作为索引值
            self.centroids = data[np.random.randint(0, data.shape[0], self.n_cluster), :]
        # 开始迭代
        for i in range(self.max_iter):
            # 1. 计算距离矩阵，得到的是一个100*6的矩阵
            distances = cdist(data, self.centroids)

            # 2.对距离按有近到远排序，选取最近的质心点的类别，作为当前点的分类
            c_ind = np.argmin(distances, axis=1)

            # 3. 对每一类数据进行均值计算，更新质心点坐标
            for i in range(self.n_cluster):
                # 排除掉没有出现在c_ind里的类别
                if i in c_ind:
                    # 选出所有类别是i的点，取data里面坐标的均值，更新第i个质心
                    self.centroids[i] = np.mean(data[c_ind == i], axis=0)

    # 实现预测方法
    def predict(self, samples):
        # 跟上面一样，先计算距离矩阵，然后选取距离最近的那个质心的类别
        distances = cdist(samples, self.centroids)
        c_ind = np.argmin(distances, axis=1)

        return c_ind
# 定义一个绘制子图函数
def plotKMeans(x, centroids, subplot, title):
    # 分配子图，121表示1行2列的子图中的第一个
    plt.subplot(subplot)
    plt.scatter(x[:, 0], x[:, 1], c='r')
    # 画出质心点
    plt.scatter(centroids[:, 0], centroids[:, 1], c=np.array(range(5)), s=100)
    plt.title(title)
if __name__ == '__main__':

    class_txt = './kmeans.txt'

    data = pd.read_csv(class_txt, header=0, sep='\t')
    x = data.iloc[:, 0:4].values
    y = data.iloc[:, -1].values
    print(x)
    print(y)
    # c = x.shape[0]
    # # print(x)
    # # print(y)
    # # print(c)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    u, v, z = x[:, 0], x[:, 1], x[:, 2]
    # print(u)
    ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
    #  将数据点分成三部分画，在颜色上有区分度
    print(u[:10])
    ax.scatter(u[:10], v[:10], z[:10], c=y)  # 绘制数据点
    ax.scatter(u[10:20], v[10:20], z[10:20], c='r')
    ax.scatter(u[30:40], v[30:40], z[30:40], c='g')

    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
    # plt.figure(figsize=(6, 6))
    # plt.scatter(x[:, 0], x[:, 2], c=y)
    # plt.show()
    # dist = np.array([[121, 221, 32, 43],
    #                  [121, 1, 12, 23],
    #                  [65, 21, 2, 43],
    #                  [1, 221, 32, 43],
    #                  [21, 11, 22, 3], ])
    # c_ind = np.argmin(dist, axis=1)
    # # print(c_ind)
    # # print(c_ind)
    # x_new = x[0:5]
    # print(x_new)
    # print(c_ind == 2)
    # print(x_new[c_ind == 2])
    # np.mean(x_new[c_ind == 2], axis=0)
    #
    # kmeans = K_Mean(max_iter=300, centroids=np.array([[2, 1], [2, 2], [2, 3], [2, 4], [2, 5]]))
    #
    # plt.figure(figsize=(16, 6))
    # plotKMeans(x, y, kmeans.centroids, 121, 'Initial State')
    #
    # # 开始聚类
    # kmeans.fit(x)
    #
    # plotKMeans(x, y, kmeans.centroids, 122, 'Final State')
    #
    # # 预测新数据点的类别
    # x_new = np.array([[0, 0], [10, 7]])
    # y_pred = kmeans.predict(x_new)
    #
    # print(kmeans.centroids)
    # print(y_pred)
    #
    # plt.scatter(x_new[:, 0], x_new[:, 1], s=100, c='black')
    #print(x)
    print()