# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/10/14 15:39 
# License: bupt
import os
import re
import numpy as np
import pandas as pd
filepath_root_true = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\正常数据\\'
filepath_root_wrong = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\异常数据\\'

real_filepath_root_true = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\正常数据-副本\\'
real_filepath_root_wrong = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\异常数据-副本\\'

filepath_root = filepath_root_wrong

class_txt = 'D:\\DeskTop\\2021年E题\\K-means数据\\kmeans.txt'


# for i in range(1, 325):
#     # 处理数据
#     # 处理数据
#     filename_read_true = '.正常.txt'
#     filename_read_wrong = '.异常.txt'
#
#     filename_read = filename_read_true
#
#     filename_read = str(i) + filename_read
#     filepath_read = real_filepath_root_true + filename_read
#
#     df = pd.read_table(filepath_read, header=None)
#     text = df.to_numpy()
#     # print(text)
#     text = np.mean(text, axis=0)
#     text = text.reshape(1, 4)
#     a = text.round(0)
#     print(a)
#     x = pd.DataFrame(a)
#     x.to_csv(class_txt, sep='\t', index=False, mode='a',
#              encoding='utf-8', header=0)

#
# for i in range(1, 325):
#     # 处理数据
#     # 处理数据
#     filename_read_true = '.正常.txt'
#     filename_read_wrong = '.异常.txt'
#
#     filename_read = filename_read_wrong
#
#     filename_read = str(i) + filename_read
#     filepath_read = real_filepath_root_wrong + filename_read
#
#     df = pd.read_table(filepath_read, header=None)
#     text = df.to_numpy()
#     # print(text)
#     text = np.mean(text, axis=0)
#     text = text.reshape(1, 4)
#     a = text.round(0)
#     print(a)
#     x = pd.DataFrame(a)
#     x.to_csv(class_txt, sep='\t', index=False, mode='a',
#              encoding='utf-8', header=0)

path1 = r'./kmeans.xlsx'
path = './kmeans.txt'
data = pd.read_excel(path1)
# print(data)
text = data.to_numpy()
print(text)
x = pd.DataFrame(text)
x.to_csv(path, sep='\t', index=False, mode= 'w',
         encoding='utf-8', header=0)
# data = pd.read_table(class_txt)
# print(data)
# data = data.to_numpy()
# print(data)
# x = pd.DataFrame(data)
# x.to_excel(class_txt1)



