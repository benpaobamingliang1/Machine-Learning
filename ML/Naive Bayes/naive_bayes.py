# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/2 22:38 
# License: bupt

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as CM

# 1.导入数据
digits = load_digits()
x, y = digits.data, digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=420)

# 2.建模，探索结果
gnb = GaussianNB().fit(x_train, y_train)

# 查看分数
acc_score = gnb.score(x_test, y_test)
print(acc_score)

# 查看预测结果
y_pred = gnb.predict(x_test)
print(y_pred)
# 查看预测的概率结果
prob = gnb.predict_proba(x_test)
print(prob)
print(prob.shape)
prob.shape  # 每一列对应一个标签下的概率
prob[1, :].sum()  # 每一行的和都是一
prob.sum(axis=1)

print(CM(y_test, y_pred))
