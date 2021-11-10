# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/7 12:25 
# License: bupt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine

wine = load_wine()

# print(wine.data)
# print(wine.target)
# 实例化
# 训练集带入实例化的模型去进行训练，使用的接口是fit
# 使用其他接口将测试集导入我们训练好的模型，去获取我们希望过去的结果（score.Y_test）
from sklearn.model_selection import train_test_split

Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)

clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
clf = clf.fit(Xtrain, Ytrain)
rfc = rfc.fit(Xtrain, Ytrain)
score_c = clf.score(Xtest, Ytest)
score_r = rfc.score(Xtest, Ytest)

print("Single Tree:{}".format(score_c)
      , "Random Forest:{}".format(score_r)
      )

# 目的是带大家复习一下交叉验证
# 交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)

clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)

plt.plot(range(1, 11), rfc_s, label="RandomForest")
plt.plot(range(1, 11), clf_s, label="Decision Tree")
plt.legend()
plt.show()

# ====================一种更加有趣也更简单的写法===================#

# 目的是带大家复习一下交叉验证
# 交叉验证：是数据集划分为n分，依次取每一份做测试集，每n-1份做训练集，多次训练模型以观测模型稳定性的方法

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)

clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)

plt.plot(range(1, 11), rfc_s, label="RandomForest")
plt.plot(range(1, 11), clf_s, label="Decision Tree")
plt.legend()
# plt.show()

# ====================一种更加有趣也更简单的写法===================#

rfc_l = []
clf_l = []

for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
    rfc_l.append(rfc_s)
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf, wine.data, wine.target, cv=10).mean()
    clf_l.append(clf_s)

plt.plot(range(1, 11), rfc_l, label="Random Forest")
plt.plot(range(1, 11), clf_l, label="Decision Tree")
plt.legend()
# plt.show()

# 是否有注意到，单个决策树的波动轨迹和随机森林一致？
# 再次验证了我们之前提到的，单个决策树的准确率越高，随机森林的准确率也会越高


#####【TIME WARNING: 2mins 30 seconds】#####

superpa = []
# for i in range(200):
#     rfc = RandomForestClassifier(n_estimators=i + 1, n_jobs=-1)
#     rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
#     superpa.append(rfc_s)
# print(max(superpa), superpa.index(max(superpa)) + 1)  # 打印出：最高精确度取值，max(superpa))+1指的是森林数目的数量n_estimators
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 201), superpa)
# plt.show()


rfc = RandomForestClassifier(n_estimators=20, random_state=2)
rfc = rfc.fit(Xtrain, Ytrain)

# 随机森林的重要属性之一：estimators，查看森林中树的状况
rfc.estimators_[0].random_state

# for i in range(len(rfc.estimators_)):
#    print(rfc.estimators_[i].random_state)

# 无需划分训练集和测试集
#
rfc = RandomForestClassifier(n_estimators=25, oob_score=True)  # 默认为False
rfc = rfc.fit(wine.data, wine.target)

# 重要属性oob_score_
rfc.oob_score_  # 0.9719101123595506

# 大家可以分别取尝试一下这些属性和接口

rfc = RandomForestClassifier(n_estimators=25)
rfc = rfc.fit(Xtrain, Ytrain)
rfc.score(Xtest, Ytest)

rfc.feature_importances_  # 结合zip可以对照特征名字查看特征重要性，参见上节决策树
rfc.apply(Xtest)  # apply返回每个测试样本所在的叶子节点的索引
rfc.predict(Xtest)  # predict返回每个测试样本的分类/回归结果
rfc.predict_proba(Xtest)

import numpy as np
from scipy.special import comb

x = np.linspace(0, 1, 20)

y = []
for epsilon in np.linspace(0, 1, 20):
    E = np.array([comb(25, i) * (epsilon ** i) * ((1 - epsilon) ** (25 - i)) for i in range(13, 26)]).sum()
    y.append(E)
plt.plot(x, y, "o-", label="when estimators are different")
plt.plot(x, x, "--", color="red", label="if all estimators are same")
plt.xlabel("individual estimator's error")
plt.ylabel("RandomForest's error")
plt.legend()
# plt.show()


# 2.随机森林在乳腺癌数据集上的调参
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = load_breast_cancer()

data

data.data.shape

data.target

# 可以看到，乳腺癌数据集有569条记录，30个特征，
# 单看维度虽然不算太高，但是样本量非常少。过拟合的情况可能存在。

rfc = RandomForestClassifier(n_estimators=100, random_state=90)
score_pre = cross_val_score(rfc, data.data, data.target, cv=10).mean()  # 交叉验证的分类默认scoring='accuracy'

score_pre

# 这里可以看到，随机森林在乳腺癌数据上的表现本就还不错，
# 在现实数据集上，基本上不可能什么都不调就看到95%以上的准确率

scorel = []
for i in range(0, 200, 10):
    rfc = RandomForestClassifier(n_estimators=i + 1,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    scorel.append(score)
print(max(scorel), (scorel.index(max(scorel)) * 10) + 1)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 201, 10), scorel)
plt.show()

# list.index([object])
# 返回这个object在列表list中的索引

scorel = []
for i in range(35, 45):
    rfc = RandomForestClassifier(n_estimators=i,
                                 n_jobs=-1,
                                 random_state=90)
    score = cross_val_score(rfc, data.data, data.target, cv=10).mean()
    scorel.append(score)
print(max(scorel), ([*range(35, 45)][scorel.index(max(scorel))]))
plt.figure(figsize=[20, 5])
plt.plot(range(35, 45), scorel)
plt.show()

# 调整max_depth

param_grid = {'max_depth': np.arange(1, 20, 1)}

#   一般根据数据的大小来进行一个试探，乳腺癌数据很小，所以可以采用1~10，或者1~20这样的试探
#   但对于像digit recognition那样的大型数据来说，我们应该尝试30~50层深度（或许还不足够
#   更应该画出学习曲线，来观察深度对模型的影响

rfc = RandomForestClassifier(n_estimators=39
                             , random_state=90
                             )
GS = GridSearchCV(rfc, param_grid, cv=10)  # 网格搜索
GS.fit(data.data, data.target)

print(GS.best_params_)

GS.best_score_  # 返回调整好的最佳参数对应的准确率

# 调整min_samples_leaf

param_grid = {'min_samples_leaf': np.arange(1, 1 + 10, 1)}

# 对于min_samples_split和min_samples_leaf,一般是从他们的最小值开始向上增加10或20
# 面对高维度高样本量数据，如果不放心，也可以直接+50，对于大型数据，可能需要200~300的范围
# 如果调整的时候发现准确率无论如何都上不来，那可以放心大胆调一个很大的数据，大力限制模型的复杂度

rfc = RandomForestClassifier(n_estimators=39
                             , random_state=90
                             )
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)

GS.best_params_

GS.best_score_

# 调整min_samples_split

param_grid = {'min_samples_split': np.arange(2, 2 + 20, 1)}

rfc = RandomForestClassifier(n_estimators=39
                             , random_state=90
                             )
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)

GS.best_params_

GS.best_score_

# 调整Criterion

param_grid = {'criterion': ['gini', 'entropy']}

rfc = RandomForestClassifier(n_estimators=39
                             , random_state=90
                             )
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data.data, data.target)

GS.best_params_

GS.best_score_