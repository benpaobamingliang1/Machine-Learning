# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/10/14 15:39 
# License: bupt
import os
import re
import numpy as np
from pasta.augment import inline
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

filepath_root_true = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\正常数据\\'
filepath_root_wrong = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\异常数据\\'

real_filepath_root_true = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\正常数据-副本\\'
real_filepath_root_wrong = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\异常数据-副本\\'

filepath_root = real_filepath_root_true

for i in range(1,325):
    # 处理数据
    filename_read_true = '.正常.txt'
    filename_read_wrong = '.异常.txt'

    filename_read = filename_read_true

    filename_read = str(i) + filename_read
    filepath_read = filepath_root + filename_read

    df_news = pd.read_table(filepath_read, header=None)
    print(df_news)
    x = df_news.mean()
    print(x)
    # # 去除异常值
    # list = []
    # list = np.mean(uniques, axis=1)
    # list = list.reshape(len(list),1)
    # print(list)
    # # 异常值分析
    # #（1）3σ原则：如果数据服从正态分布，异常值被定义为一组测定值中与平均值的偏差超过3倍的值 → p(|x - μ| > 3σ) ≤ 0.003
    # u = list.mean()  # 计算均值
    # std = list.std()  # 计算标准差
    # stats.kstest(list, 'norm', (u, std))
    # print('均值为：%.3f，标准差为：%.3f' % (u, std))
    # 图表表达
    # 写入数据