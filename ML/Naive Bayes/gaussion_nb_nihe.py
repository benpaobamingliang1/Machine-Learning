# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/3 10:37 
# License: bupt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from time import time
import datetime


# 1.绘制学习曲线的函数
def plot_learning_curve(estimator, title, X, y,
                        ax,  # 选择子图
                        ylim=None,  # 设置纵坐标的取值范围
                        cv=None,  # 交叉验证
                        n_jobs=None  # 设定需要使用的线程
                        ):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            , cv=cv, n_jobs=n_jobs)
    ax.title(title)
    # if ylim is not None:
       # ax.set_ylim(*ylim)
    ax.xlabel("Training examples")
    ax.ylabel("Score")
    ax.grid()  # 显示网格作为背景，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
            , color="r", label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
            , color="g", label="Test score")
    ax.legend(loc="best")
    plt.show()

# 2. 导入数据 定义循环
digits = load_digits()
X, y = digits.data, digits.target
X.shape
X  # 是一个稀疏矩阵
title = ["Naive Bayes", "DecisionTree", "SVM, RBF kernel", "RandomForest", "Logistic"]
model = [GaussianNB(), DTC(), SVC(gamma=0.001)
    , RFC(n_estimators=50), LR(C=.1, solver="lbfgs")]
cv = ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)

# 3. 进入循环，绘制学习曲线
# ax = plt.figure(figsize=(6,6))
# fig, axes = plt.subplots(1, 1, figsize=(6, 6))
# print( title, model)
print(range(len(title)))
for ind, title_, estimator in zip(range(len(title)), title, model):
    times = time()

    plot_learning_curve(estimator, title_, X, y,
                        plt, ylim=[0.7, 1.05], n_jobs=4, cv=cv)
    print("{}:{}".format(title_, datetime.datetime.fromtimestamp(time() -
                                                                 times).strftime("%M:%S:%f")))
