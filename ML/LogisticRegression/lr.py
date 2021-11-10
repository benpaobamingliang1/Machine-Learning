# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/3 16:27 
# License: bupt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 精确性分数

data = load_breast_cancer()  # 乳腺癌数据集
X = data.data
y = data.target

# print(X, y)
X.data.shape  # (569, 30)

lrl1 = LR(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)

lrl2 = LR(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)

# 逻辑回归的重要属性coef_，查看每个特征所对应的参数
lrl1 = lrl1.fit(X, y)
print(lrl1.coef_)
(lrl1.coef_ != 0).sum(axis=1)  # array([10])    30个特征中有10个特征的系数不为0

lrl2 = lrl2.fit(X, y)
print(lrl2.coef_)

from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 精确性分数

data = load_breast_cancer()  # 乳腺癌数据集
X = data.data
y = data.target

X.data.shape  # (569, 30)

lrl1 = LR(penalty="l1", solver="liblinear", C=0.5, max_iter=1000)

lrl2 = LR(penalty="l2", solver="liblinear", C=0.5, max_iter=1000)

# 逻辑回归的重要属性coef_，查看每个特征所对应的参数
lrl1 = lrl1.fit(X, y)
lrl1.coef_

(lrl1.coef_ != 0).sum(axis=1)  # array([10])    30个特征中有10个特征的系数不为0

lrl2 = lrl2.fit(X, y)
lrl2.coef_

# 2.逻辑斯蒂回归
l1 = []
l2 = []
l1test = []
l2test = []

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

for i in np.linspace(0.05, 1.5, 19):
    lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
    lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)

    lrl1 = lrl1.fit(Xtrain, Ytrain)
    l1.append(accuracy_score(lrl1.predict(Xtrain), Ytrain))
    l1test.append(accuracy_score(lrl1.predict(Xtest), Ytest))
    lrl2 = lrl2.fit(Xtrain, Ytrain)
    l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
    l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))

graph = [l1, l2, l1test, l2test]
color = ["green", "black", "lightgreen", "gray"]
label = ["L1", "L2", "L1test", "L2test"]

plt.figure(figsize=(6, 6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05, 1.5, 19), graph[i], color[i], label=label[i])
plt.legend(loc=0)  # 图例的位置在哪里?4表示，右下角
plt.show()


# 使用 高效的嵌入法embedded对逻辑斯蒂回回模型进行降维
from sklearn.linear_model import LogisticRegression as LR
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel

data = load_breast_cancer()
data.data.shape

LR_ = LR(solver="liblinear", C=0.9, random_state=420)
cross_val_score(LR_, data.data, data.target, cv=10).mean()

X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)

X_embedded.shape  # (569, 9)

cross_val_score(LR_, X_embedded, data.target, cv=10).mean()  # 0.9368323826808401

# 3.2 通过调整阈值的方法对embedding降维
fullx = []
fsx = []

threshold = np.linspace(0, abs((LR_.fit(data.data, data.target).coef_)).max(), 20)
print(threshold)
k = 0
for i in threshold:
    X_embedded = SelectFromModel(LR_, threshold=i).fit_transform(data.data, data.target)
    fullx.append(cross_val_score(LR_, data.data, data.target, cv=5).mean())
    fsx.append(cross_val_score(LR_, X_embedded, data.target, cv=5).mean())
    print((threshold[k], X_embedded.shape[1]))
    k += 1

plt.figure(figsize=(20, 5))
plt.plot(threshold, fullx, label="full")
plt.plot(threshold, fsx, label="feature selection")
plt.xticks(threshold)
plt.legend()
plt.show()

# 3.3 第二种调整方法，是调逻辑回归的类LR_，通过画C的学习曲线来实现：
fullx = []
fsx = []

C = np.arange(0.01, 10.01, 0.5)

for i in C:
    LR_ = LR(solver="liblinear", C=i, random_state=420)

    fullx.append(cross_val_score(LR_, data.data, data.target, cv=10).mean())

    X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)
    fsx.append(cross_val_score(LR_, X_embedded, data.target, cv=10).mean())

print(max(fsx), C[fsx.index(max(fsx))])

plt.figure(figsize=(20, 5))
plt.plot(C, fullx, label="full")
plt.plot(C, fsx, label="feature selection")
plt.xticks(C)
plt.legend()
plt.show()


# 3.4 完整性测试
fullx = []
fsx = []

C = np.arange(6.05, 7.05, 0.005)

for i in C:
    LR_ = LR(solver="liblinear", C=i, random_state=420)

    fullx.append(cross_val_score(LR_, data.data, data.target, cv=10).mean())

    X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)
    fsx.append(cross_val_score(LR_, X_embedded, data.target, cv=10).mean())

print(max(fsx), C[fsx.index(max(fsx))])

plt.figure(figsize=(20, 5))
plt.plot(C, fullx, label="full")
plt.plot(C, fsx, label="feature selection")
plt.xticks(C)
plt.legend()
plt.show()

# 验证模型效果：降维之前
LR_ = LR(solver="liblinear", C=6.069999999999999, random_state=420)
cross_val_score(LR_, data.data, data.target, cv=10).mean()  # 0.947360859044162

# 验证模型效果：降维之后
LR_ = LR(solver="liblinear", C=6.069999999999999, random_state=420)
X_embedded = SelectFromModel(LR_, norm_order=1).fit_transform(data.data, data.target)
cross_val_score(LR_, X_embedded, data.target, cv=10).mean()  # 0.9580405755768732

X_embedded.shape  # (569, 10)


# 3.5 l2测试
l2 = []
l2test = []

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

for i in np.arange(1, 201, 10):
    lrl2 = LR(penalty="l2", solver="liblinear", C=0.9, max_iter=i)
    lrl2 = lrl2.fit(Xtrain, Ytrain)
    l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
    l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))

graph = [l2, l2test]
color = ["black", "gray"]
label = ["L2", "L2test"]

plt.figure(figsize=(20, 5))
for i in range(len(graph)):
    plt.plot(np.arange(1, 201, 10), graph[i], color[i], label=label[i])
plt.legend(loc=4)
plt.xticks(np.arange(1, 201, 10))
plt.show()

# 我们可以使用属性.n_iter_来调用本次求解中真正实现的迭代次数

lr = LR(penalty="l2", solver="liblinear", C=0.9, max_iter=300).fit(Xtrain, Ytrain)
lr.n_iter_  # array([24], dtype=int32)  只迭代了24次就达到收敛


# 3.6 测试
from sklearn.datasets import load_iris

iris = load_iris()
iris.target  # 三分类数据集

for multi_class in ('multinomial', 'ovr'):
    clf = LR(solver='sag', max_iter=100, random_state=42,
             multi_class=multi_class).fit(iris.data, iris.target)

    # 打印两种multi_class模式下的训练分数
    # %的用法，用%来代替打印的字符串中，想由变量替换的部分。%.3f表示，保留三位小数的浮点数。%s表示，字符串。
    # 字符串后的%后使用元祖来容纳变量，字符串中有几个%，元祖中就需要有几个变量

    print("training score : %.3f (%s)" % (clf.score(iris.data, iris.target), multi_class))