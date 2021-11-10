# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/10/14 15:39 
# License: bupt
import os
import re
import numpy as np
import pandas as pd
path_dir = os.listdir('D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\正常数据')
path_name = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\正常数据\\'

path_name1 = 'D:\\DeskTop\\2021年E题\\附件1：UWB数据集\\正常数据-副本\\'
i = 1
for file in path_dir:
    file_name = path_name + file
    normal_txt = open(file_name, 'r', encoding='utf-8')
    text = []
    for index, str_line in enumerate(normal_txt.readlines()):
        if index >= 1:
            strings = str_line.split(':')
            if index % 4 == 1:
                a = strings[1]
                text = np.append(text, a)
            b = strings[5]
            text = np.append(text, b)
    text = text.reshape(int((index) / 4), 5)
    np.set_printoptions(threshold=np.inf)
    print(text)
    print(file)
    normal_txt.close()


    #写入数据
    filename_write = '.正常.txt'
    filename_write = str(i) + filename_write
    filepath_write = path_name1 + filename_write
    print(filepath_write)
    fw = open(filepath_write, 'w', encoding='utf-8')
    text = str(text)
    fw.write(text)
    fw.close()
    i = i + 1