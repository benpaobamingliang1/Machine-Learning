# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/10/11 15:56 
# License: bupt
# -*- coding:utf-8 -*-
# 导入鸢尾花数据集，调用matplotlib包用于数据的可视化，并加载PCA算法包。
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# 然后以字典的形式加载鸢尾花数据集，使用y表示数据集中的标签，使用x表示数据集中的属性数据。
data = load_iris()
y = data.target
x = data.data
print(y)
print(x)

# 将数据写入 excel 文件
def data_write_excel(data, filename, writer=None):
 data = pd.DataFrame(data)
 if not writer:
     writer = pd.ExcelWriter(filename + '1.xlsx')
     data.to_excel(writer, float_format='%.5f', sheet_name=filename)
     writer.save()
     writer.close()
 else:
    data.to_excel(writer, float_format='%.5f', sheet_name=filename)
# 调用PCA算法进行降维主成分分析
# 指定主成分个数，即降维后数据维度，降维后的数据保存在reduced_x中。
pca = PCA(n_components=2)
reduced_x = pca.fit_transform(x)
writer = pd.ExcelWriter('pca1.xlsx')
data_restore = pca.inverse_transform(reduced_x)
data_write_excel(reduced_x, 'low', writer=writer)
data_write_excel(data_restore, 'restore', writer=writer)
# 主成分贡献率
data_write_excel(pca.explained_variance_ratio_, 'ratio', writer=writer)
# 主成分方差
data_write_excel(pca.explained_variance_, 'variance', writer=writer)
# 主成分在各个变量的负载
data_write_excel(pca.components_.T, 'component', writer=writer)
# 主成分个数
print(pca.n_components_, ' n_components')
writer.save()
writer.close()

print(data_restore)
# 将降维后的数据保存在不同的列表中
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])

    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])

    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

# 可视化
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()