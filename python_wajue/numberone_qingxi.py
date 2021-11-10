# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/10/10 17:48 
# License: bupt
import numpy as np
import pandas as pd
import sys
import re

# f1 = open('./test1.csv', 'r+')
# f2 = open('./test2.csv', 'w+')
# str1 = r'-'
# str2 = r','
# for ss in f1.readlines():
#     tt = re.sub(str1, str2, ss)
#     f2.write(tt)
# f1.close()
# f2.close()
#1.读取文件

points = np.genfromtxt('test2.csv', delimiter=',')
#提取points中的两列数据，分别作为min,max
# x1 = points[:, 0]
# x2 = points[:, 1]
# print(x1)
# print(x2)
# print(points.shape)

points1 = np.genfromtxt('test3.csv', delimiter=',')
num1 = {}
print(points1)
for i,(x0,x1) in enumerate(points):
    num1[i] = 0
    for j,y in enumerate(points1):
        if ( y[i] < x0 or y[i] >x1 ):
            num1[i] = int(num1[i]) + 1
data1 = pd.DataFrame(points1)
data1.to_csv('data1.csv')

#相关性分析

