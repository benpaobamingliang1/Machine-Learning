# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/1 16:10 
# License: bupt
import numpy  as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection

from sklearn import tree
from sklearn.datasets import load_wine
import graphviz
import matplotlib.pyplot as plt

# 2.导入需要的算法库和模块
wine = load_wine()
# print(type(wine))

x = wine.data
y = wine.target
# 如果wine是一张表，应该长这样：
# pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
print(wine.feature_names)
# wine.target_names

y = y.reshape(-1, 1)
# 2.划分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=50, stratify=y)

print(x_train.shape, y_train.shape)

# 3. 建立模型
clf = tree.DecisionTreeClassifier(criterion="gini"
                                  , random_state=30
                                  , splitter="random")
clf = clf.fit(x_train, y_train)
# 返回预测的准确度
score = clf.score(x_test, y_test)
print(score)
feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素'
    , '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']

dot_data = tree.export_graphviz(clf
                                , feature_names=feature_name
                                , class_names=["琴酒", "雪莉", "贝尔摩德"]
                                , filled=True
                                , rounded=True
                                )
graph = graphviz.Source(dot_data)
# print(graph)

print(clf.feature_importances_)
print(list(zip(feature_name, clf.feature_importances_)))

# 4. 剪枝函数
# 我们的树对训练集的拟合程度如何？,解决过拟合
score_train = clf.score(x_train, y_train)
print(score_train)

clf = tree.DecisionTreeClassifier(criterion="gini"
                                  , random_state=30
                                  , splitter="best"
                                  , max_depth=3
                                  , min_samples_leaf=10
                                  , min_samples_split=10
                                  )
clf = clf.fit(x_train, y_train)
# 返回预测的准确度
score = clf.score(x_test, y_test)
score_train = clf.score(x_train, y_train)
print(score_train)
print(score)
feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素'
    , '颜色强度', '色调', 'od280/od315稀释葡萄酒', '脯氨酸']

# 5. 可视化参数
test = []
for i in range(10):
    clf = tree.DecisionTreeClassifier(max_depth=i + 1
                                      , criterion="entropy"
                                      , random_state=30
                                      , splitter="random"
                                      )
    clf = clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    test.append(score)
plt.plot(range(1, 11), test, color="red", label="max_depth")
plt.legend()
plt.show()
