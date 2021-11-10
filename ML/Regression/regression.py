# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/9 20:01 
# License: bupt


# 1. 导库
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing as fch  # 加利福尼亚房屋价值数据集
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

housevalue = fch()  # 会需要下载，大家可以提前运行试试看
housevalue.data

X = pd.DataFrame(housevalue.data)

y = housevalue.target

# print(y.min())
#
# print(y.max())

X.columns = housevalue.feature_names

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=0.3, random_state=420)

# 恢复索引
for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

reg = LR().fit(Xtrain, Ytrain)

yhat = reg.predict(Xtest)  # 预测我们的yhat

[*zip(Xtrain.columns, reg.coef_)]

# 截距项
reg.intercept_

from sklearn.metrics import mean_squared_error as MSE

MSE(yhat, Ytest)
# 会报错，因为在虽然均方误差永远为正，但是sklearn中的参数scoring下，均方误差作为评判
# 标准时，却是计算”负均方误差“（neg_mean_squared_error）。
# cross_val_score(reg, X, y, cv=10, scoring="mean_squared_error")

print(cross_val_score(reg, X, y, cv=10, scoring="neg_mean_squared_error").mean())

print(cross_val_score(reg, X, y, cv=10, scoring="r2").mean())

import matplotlib.pyplot as plt

sorted(Ytest)
plt.plot(range(len(Ytest)), sorted(Ytest), c="black", label="Data")
plt.plot(range(len(yhat)), sorted(yhat), c="red", label="Predict")
plt.legend()
plt.show()

import numpy as np

rng = np.random.RandomState(42)
X = rng.randn(100, 80)
y = rng.randn(100)
cross_val_score(LR(), X, y, cv=5, scoring='r2').mean()

# 2.引入加利福尼亚州数据
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split as TTS
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt

housevalue = fch()

X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目"
    , "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

# 使用岭回归来进行建模
reg = Ridge(alpha=1).fit(Xtrain, Ytrain)
reg.score(Xtest, Ytest)  # 加利佛尼亚房屋价值数据集中应该不是共线性问题

# 交叉验证下，与线性回归相比，岭回归的结果如何变化？
alpharange = np.arange(1, 1001, 100)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg, X, y, cv=5, scoring="r2").mean()
    linears = cross_val_score(linear, X, y, cv=5, scoring="r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpharange, ridge, color="red", label="Ridge")
plt.plot(alpharange, lr, color="orange", label="LR")
plt.title("Mean")
plt.legend()
plt.show()

# 使用岭回归来进行建模
reg = Ridge(alpha=0).fit(Xtrain, Ytrain)
reg.score(Xtest, Ytest)  # 加利佛尼亚房屋价值数据集中应该不是共线性问题

# 细化一下学习曲线
alpharange = np.arange(1, 201, 10)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg, X, y, cv=5, scoring="r2").mean()
    linears = cross_val_score(linear, X, y, cv=5, scoring="r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpharange, ridge, color="red", label="Ridge")
plt.plot(alpharange, lr, color="orange", label="LR")
plt.title("Mean")
plt.legend()
plt.show()

# 模型方差如何变化？
alpharange = np.arange(1, 1001, 100)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    varR = cross_val_score(reg, X, y, cv=5, scoring="r2").var()
    varLR = cross_val_score(linear, X, y, cv=5, scoring="r2").var()
    ridge.append(varR)
    lr.append(varLR)
plt.plot(alpharange, ridge, color="red", label="Ridge")
plt.plot(alpharange, lr, color="orange", label="LR")
plt.title("Variance")
plt.legend()
plt.show()

# 探究波士顿
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score

X = load_boston().data
y = load_boston().target

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

# 先查看方差的变化
alpharange = np.arange(1, 1001, 100)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    varR = cross_val_score(reg, X, y, cv=5, scoring="r2").var()
    varLR = cross_val_score(linear, X, y, cv=5, scoring="r2").var()
    ridge.append(varR)
    lr.append(varLR)
plt.plot(alpharange, ridge, color="red", label="Ridge")
plt.plot(alpharange, lr, color="orange", label="LR")
plt.title("Variance")
plt.legend()
plt.show()

# 查看R2的变化
alpharange = np.arange(1, 1001, 100)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    linear = LinearRegression()
    regs = cross_val_score(reg, X, y, cv=5, scoring="r2").mean()
    linears = cross_val_score(linear, X, y, cv=5, scoring="r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpharange, ridge, color="red", label="Ridge")
plt.plot(alpharange, lr, color="orange", label="LR")
plt.title("Mean")
plt.legend()
plt.show()

# 细化学习曲线
alpharange = np.arange(100, 300, 10)
ridge, lr = [], []
for alpha in alpharange:
    reg = Ridge(alpha=alpha)
    # linear = LinearRegression()
    regs = cross_val_score(reg, X, y, cv=5, scoring="r2").mean()
    # linears = cross_val_score(linear,X,y,cv=5,scoring = "r2").mean()
    ridge.append(regs)
    lr.append(linears)
plt.plot(alpharange, ridge, color="red", label="Ridge")
# plt.plot(alpharange,lr,color="orange",label="LR")
plt.title("Mean")
plt.legend()
plt.show()

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split as TTS
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt

housevalue = fch()

X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目"
    , "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]

Ridge_ = RidgeCV(alphas=np.arange(1, 1001, 100)
                 # ,scoring="neg_mean_squared_error"
                 , store_cv_values=True
                 # ,cv=5
                 ).fit(X, y)

# 无关交叉验证的岭回归结果
Ridge_.score(X, y)

# 调用所有交叉验证的结果
Ridge_.cv_values_

# 进行平均后可以查看每个正则化系数取值下的交叉验证结果
Ridge_.cv_values_.mean(axis=0)

# 查看被选择出来的最佳正则化系数
Ridge_.alpha_

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split as TTS
from sklearn.datasets import fetch_california_housing as fch
import matplotlib.pyplot as plt

housevalue = fch()

X = pd.DataFrame(housevalue.data)
y = housevalue.target
X.columns = ["住户收入中位数", "房屋使用年代中位数", "平均房间数目"
    , "平均卧室数目", "街区人口", "平均入住率", "街区的纬度", "街区的经度"]

X.head()

Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)

for i in [Xtrain, Xtest]:
    i.index = range(i.shape[0])

# 线性回归进行拟合
reg = LinearRegression().fit(Xtrain, Ytrain)
(reg.coef_ * 100).tolist()

# 岭回归进行拟合
Ridge_ = Ridge(alpha=0).fit(Xtrain, Ytrain)
(Ridge_.coef_ * 100).tolist()

# # Lasso进行拟合
# lasso_ = Lasso(alpha=0).fit(Xtrain, Ytrain)
# (lasso_.coef_ * 100).tolist()

# 岭回归进行拟合
Ridge_ = Ridge(alpha=0.1).fit(Xtrain, Ytrain)
(Ridge_.coef_ * 100).tolist()

# Lasso进行拟合
lasso_ = Lasso(alpha=0.1).fit(Xtrain, Ytrain)
(lasso_.coef_ * 100).tolist()

lasso_ = Lasso(alpha=10 ** 4).fit(Xtrain, Ytrain)
(lasso_.coef_ * 100).tolist()

# 加大正则项系数，观察模型的系数发生了什么变化
Ridge_ = Ridge(alpha=10 ** 10).fit(Xtrain, Ytrain)
(Ridge_.coef_ * 100).tolist()

# 将系数进行绘图
plt.plot(range(1, 9), (reg.coef_ * 100).tolist(), color="red", label="LR")
plt.plot(range(1, 9), (Ridge_.coef_ * 100).tolist(), color="orange", label="Ridge")
plt.plot(range(1, 9), (lasso_.coef_ * 100).tolist(), color="k", label="Lasso")
plt.plot(range(1, 9), [0] * 8, color="grey", linestyle="--")
plt.xlabel('w')  # 横坐标是每一个特征所对应的系数
plt.legend()
plt.show()

#
from sklearn.linear_model import LassoCV

# 自己建立Lasso进行alpha选择的范围
alpharange = np.logspace(-10, -2, 200, base=10)

# 其实是形成10为底的指数函数
# 10**(-10)到10**(-2)次方

alpharange.shape

Xtrain.head()

lasso_ = LassoCV(alphas=alpharange  # 自行输入的alpha的取值范围
                 , cv=5  # 交叉验证的折数
                 ).fit(Xtrain, Ytrain)

# 查看被选择出来的最佳正则化系数
lasso_.alpha_

# 调用所有交叉验证的结果
lasso_.mse_path_

lasso_.mse_path_.shape  # 返回每个alpha下的五折交叉验证结果

lasso_.mse_path_.mean(axis=1)  # 有注意到在岭回归中我们的轴向是axis=0吗？

# 在岭回归当中，我们是留一验证，因此我们的交叉验证结果返回的是，每一个样本在每个alpha下的交叉验证结果
# 因此我们要求每个alpha下的交叉验证均值，就是axis=0，跨行求均值
# 而在这里，我们返回的是，每一个alpha取值下，每一折交叉验证的结果
# 因此我们要求每个alpha下的交叉验证均值，就是axis=1，跨列求均值

# 最佳正则化系数下获得的模型的系数结果
lasso_.coef_

lasso_.score(Xtest, Ytest)

# 与线性回归相比如何？
reg = LinearRegression().fit(Xtrain, Ytrain)
reg.score(Xtest, Ytest)

# 使用lassoCV自带的正则化路径长度和路径中的alpha个数来自动建立alpha选择的范围
ls_ = LassoCV(eps=0.00001
              , n_alphas=300
              , cv=5
              ).fit(Xtrain, Ytrain)

ls_.alpha_

ls_.alphas_  # 查看所有自动生成的alpha取值

ls_.alphas_.shape

ls_.score(Xtest, Ytest)

ls_.coef_

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

rnd = np.random.RandomState(42)  # 设置随机数种子
X = rnd.uniform(-3, 3, size=100)  # random.uniform，从输入的任意两个整数中取出size个随机数

# 生成y的思路：先使用NumPy中的函数生成一个sin函数图像，然后再人为添加噪音
y = np.sin(X) + rnd.normal(size=len(X)) / 3  # random.normal，生成size个服从正态分布的随机数
# 使用散点图观察建立的数据集是什么样子
plt.scatter(X, y, marker='o', c='k', s=20)
plt.show()

X = X.reshape(-1, 1)

# 使用原始数据进行建模
LinearR = LinearRegression().fit(X, y)
TreeR = DecisionTreeRegressor(random_state=0).fit(X, y)

# 放置画布
fig, ax1 = plt.subplots(1)

# 创建测试数据：一系列分布在横坐标上的点
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

# 将测试数据带入predict接口，获得模型的拟合效果并进行绘制
ax1.plot(line, LinearR.predict(line), linewidth=2, color='green',
         label="linear regression")
ax1.plot(line, TreeR.predict(line), linewidth=2, color='red',
         label="decision tree")

# 将原数据上的拟合绘制在图像上
ax1.plot(X[:, 0], y, 'o', c='k')

# 其他图形选项
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Result before discretization")
plt.tight_layout()
plt.show()

# 从这个图像来看，可以得出什么结果？

# 准备数据
enc = KBinsDiscretizer(n_bins=10, encode="onehot")
X_binned = enc.fit_transform(X)
line_binned = enc.transform(line)

# 将两张图像绘制在一起，布置画布
fig, (ax1, ax2) = plt.subplots(ncols=2
                               , sharey=True  # 让两张图共享y轴上的刻度
                               , figsize=(10, 4))

# 在图1中布置在原始数据上建模的结果
ax1.plot(line, LinearR.predict(line), linewidth=2, color='green',
         label="linear regression")
ax1.plot(line, TreeR.predict(line), linewidth=2, color='red',
         label="decision tree")
ax1.plot(X[:, 0], y, 'o', c='k')
ax1.legend(loc="best")
ax1.set_ylabel("Regression output")
ax1.set_xlabel("Input feature")
ax1.set_title("Result before discretization")

# 使用分箱数据进行建模
LinearR_ = LinearRegression().fit(X_binned, y)
TreeR_ = DecisionTreeRegressor(random_state=0).fit(X_binned, y)

# 进行预测，在图2中布置在分箱数据上进行预测的结果
ax2.plot(line  # 横坐标
         , LinearR_.predict(line_binned)  # 分箱后的特征矩阵的结果
         , linewidth=2
         , color='green'
         , linestyle='-'
         , label='linear regression')

ax2.plot(line, TreeR_.predict(line_binned), linewidth=2, color='red',
         linestyle=':', label='decision tree')

# 绘制和箱宽一致的竖线
ax2.vlines(enc.bin_edges_[0]  # x轴
           , *plt.gca().get_ylim()  # y轴的上限和下限
           , linewidth=1
           , alpha=.2)

# 将原始数据分布放置在图像上
ax2.plot(X[:, 0], y, 'o', c='k')

# 其他绘图设定
ax2.legend(loc="best")
ax2.set_xlabel("Input feature")
ax2.set_title("Result after discretization")
plt.tight_layout()
plt.show()

enc = KBinsDiscretizer(n_bins=15, encode="onehot")
X_binned = enc.fit_transform(X)
line_binned = enc.transform(line)

fig, ax2 = plt.subplots(1, figsize=(5, 4))

LinearR_ = LinearRegression().fit(X_binned, y)
print(LinearR_.score(line_binned, np.sin(line)))
TreeR_ = DecisionTreeRegressor(random_state=0).fit(X_binned, y)

ax2.plot(line  # 横坐标
         , LinearR_.predict(line_binned)  # 分箱后的特征矩阵的结果
         , linewidth=2
         , color='green'
         , linestyle='-'
         , label='linear regression')
ax2.plot(line, TreeR_.predict(line_binned), linewidth=2, color='red',
         linestyle=':', label='decision tree')
ax2.vlines(enc.bin_edges_[0], *plt.gca().get_ylim(), linewidth=1, alpha=.2)
ax2.plot(X[:, 0], y, 'o', c='k')
ax2.legend(loc="best")
ax2.set_xlabel("Input feature")
ax2.set_title("Result after discretization")
plt.tight_layout()
plt.show()

# 怎样选取最优的箱子?
from sklearn.model_selection import cross_val_score as CVS
import numpy as np

pred, score, var = [], [], []
binsrange = [2, 5, 10, 15, 20, 30]
for i in binsrange:
    # 实例化分箱类
    enc = KBinsDiscretizer(n_bins=i, encode="onehot")
    # 转换数据
    X_binned = enc.fit_transform(X)
    line_binned = enc.transform(line)
    # 建立模型
    LinearR_ = LinearRegression()
    # 全数据集上的交叉验证
    cvresult = CVS(LinearR_, X_binned, y, cv=5)
    score.append(cvresult.mean())
    var.append(cvresult.var())
    # 测试数据集上的打分结果
    pred.append(LinearR_.fit(X_binned, y).score(line_binned, np.sin(line)))
# 绘制图像
plt.figure(figsize=(6, 5))
plt.plot(binsrange, pred, c="orange", label="test")
plt.plot(binsrange, score, c="k", label="full data")
plt.plot(binsrange, score + np.array(var) * 0.5, c="red", linestyle="--", label="var")
plt.plot(binsrange, score - np.array(var) * 0.5, c="red", linestyle="--")
plt.legend()
plt.show()
