# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/10 9:54 
# License: bupt

from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LinearR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS
from sklearn.metrics import mean_squared_error as MSE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
import datetime

data = load_boston()
X = data.data
y = data.target
Xtrain, Xtest, Ytrain, Ytest = TTS(X, y, test_size=0.3, random_state=420)
reg = XGBR(n_estimators=100).fit(Xtrain, Ytrain)  # 训练
reg.predict(Xtest)  # 传统接口predict

print(reg.score(Xtest, Ytest))
print(y.mean())

# 可以看出均方误差是平均值y.mean()的1/3左右，结果不算好也不算坏
print(MSE(Ytest, reg.predict(Xtest)))

# 树模型的优势之一：能够查看模型的重要性分数，可以使用嵌入法(SelectFromModel)进行特征选择
# xgboost可以使用嵌入法进行特征选择
reg.feature_importances_

# 2.验证GBDT模型与XGBoost模型的区别，交叉验证
# 交叉验证中导入的没有经过训练的模型
reg = XGBR(n_estimators=100)

# 这里应该返回什么模型评估指标，还记得么？ 返回的是与reg.score相同的评估指标R^2（回归），准确率（分类）
print("----分类准确率")
print(CVS(reg, Xtrain, Ytrain, cv=5).mean())

CVS(reg, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean()

# 来查看一下sklearn中所有的模型评估指标
import sklearn

sorted(sklearn.metrics.SCORERS.keys())

# 使用随机森林和线性回归进行一个对比
rfr = RFR(n_estimators=100)
# 0.7975497480638329
# print(CVS(rfr, Xtrain, Ytrain, cv=5).mean())

lr = LinearR()
# 0.6835070597278085
# print(CVS(lr, Xtrain, Ytrain, cv=5).mean())

# 如果开启参数slient：在数据巨大，预料到算法运行会非常缓慢的时候可以使用这个参数来监控模型的训练进度
reg = XGBR(n_estimators=10)  # xgboost库silent=True不会打印训练进程，只返回运行结果，默认是False会打印训练进程


# sklearn库中的xgbsoost的默认为silent=True不会打印训练进程，想打印需要手动设置为False
# -92.67865836936579
# print(CVS(reg, Xtrain, Ytrain, cv=5, scoring='neg_mean_squared_error').mean())


def plot_learning_curve(estimator, title, X, y,
                        ax=None,  # 选择子图
                        ylim=None,  # 设置纵坐标的取值范围
                        cv=None,  # 交叉验证
                        n_jobs=None  # 设定索要使用的线程
                        ):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y
                                                            , shuffle=True
                                                            , cv=cv
                                                            , random_state=420
                                                            , n_jobs=n_jobs)
    if ax == None:
        ax = plt.gca()
    else:
        ax = plt.figure()
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.grid()  # 绘制网格，不是必须
    ax.plot(train_sizes, np.mean(train_scores, axis=1), 'o-'
            , color="r", label="Training score")
    ax.plot(train_sizes, np.mean(test_scores, axis=1), 'o-'
            , color="g", label="Test score")
    ax.legend(loc="best")
    return ax


# 交叉验证模式
cv = KFold(n_splits=10, shuffle=True, random_state=42)

plot_learning_curve(XGBR(n_estimators=100, random_state=420)
                    , "XGB", Xtrain, Ytrain, ax=None, cv=cv)
plt.show()

plot_learning_curve(XGBR(n_estimators=100, random_state=420)
                    , "XGB", Xtrain, Ytrain, ax=None, cv=cv)
plt.show()

# =====【TIME WARNING：25 seconds】=====#

# axisx = range(10, 1010, 50)
# rs = []
# for i in axisx:
#     reg = XGBR(n_estimators=i, random_state=420)
#     rs.append(CVS(reg, Xtrain, Ytrain, cv=cv).mean())
# # print(axisx[rs.index(max(rs))], max(rs))
# plt.figure(figsize=(20, 5))
# plt.plot(axisx, rs, c="red", label="XGB")
# plt.legend()
# plt.show()

# ======【TIME WARNING: 20s】=======#
axisx = range(50, 1050, 50)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    # 记录1-偏差
    rs.append(cvresult.mean())
    # 记录方差
    var.append(cvresult.var())
    # 计算泛化误差的可控部分
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
# 打印R2最高所对应的参数取值，并打印这个参数下的方差
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
# 打印方差最低时对应的参数取值，并打印这个参数下的R2
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
# 打印泛化误差可控部分的参数取值，并打印这个参数下的R2，方差以及泛化误差的可控部分
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="red", label="XGB")
plt.legend()
# plt.show()

axisx = range(100, 300, 10)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
rs = np.array(rs)
var = np.array(var) * 0.01
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="black", label="XGB")
# 添加方差线
plt.plot(axisx, rs + var, c="red", linestyle='-.')
plt.plot(axisx, rs - var, c="red", linestyle='-.')
plt.legend()
plt.show()

# 看看泛化误差的可控部分如何？
plt.figure(figsize=(20, 5))
plt.plot(axisx, ge, c="gray", linestyle='-.')
plt.show()

# 验证模型效果是否提高了？
time0 = time()
print("弱学习器为100")
print(XGBR(n_estimators=100, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)

time0 = time()
print("弱学习器为660")
print(XGBR(n_estimators=660, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)

time0 = time()
print(XGBR(n_estimators=180, random_state=420).fit(Xtrain, Ytrain).score(Xtest, Ytest))
print(time() - time0)

# 2. 调整继续细化学习曲线,调整subsampled的大小
axisx = np.linspace(0.05, 1, 20)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=180, subsample=i, random_state=420)
    cvresult = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(cvresult.mean())
    var.append(cvresult.var())
    ge.append((1 - cvresult.mean()) ** 2 + cvresult.var())
print("-------subsample的大小--------")
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
rs = np.array(rs)
var = np.array(var)
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="black", label="XGB")
plt.plot(axisx, rs + var, c="red", linestyle='-.')
plt.plot(axisx, rs - var, c="red", linestyle='-.')
plt.legend()
plt.show()


# 首先我们先来定义一个评分函数，这个评分函数能够帮助我们直接打印Xtrain上的交叉验证结果
def regassess(reg, Xtrain, Ytrain, cv, scoring=["r2"], show=True):
    score = []
    for i in range(len(scoring)):
        if show:
            print("{}:{:.2f}".format(scoring[i]  # 模型评估指标的名字
                                     , CVS(reg
                                           , Xtrain, Ytrain
                                           , cv=cv, scoring=scoring[i]).mean()))
        score.append(CVS(reg, Xtrain, Ytrain, cv=cv, scoring=scoring[i]).mean())
    return score


from time import time
import datetime

for i in [0, 0.2, 0.5, 1]:
    time0 = time()
    reg = XGBR(n_estimators=180, random_state=420, learning_rate=i)
    print("learning_rate = {}".format(i))
    regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"])
    print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))
    print("\t")

axisx = np.arange(0.05, 1, 0.05)
rs = []
te = []
for i in axisx:
    reg = XGBR(n_estimators=180, random_state=420, learning_rate=i)
    score = regassess(reg, Xtrain, Ytrain, cv, scoring=["r2", "neg_mean_squared_error"], show=False)
    test = reg.fit(Xtrain, Ytrain).score(Xtest, Ytest)
    rs.append(score[0])
    te.append(test)
print(axisx[rs.index(max(rs))], max(rs))
plt.figure(figsize=(20, 5))
plt.plot(axisx, te, c="gray", label="test")
plt.plot(axisx, rs, c="green", label="train")
plt.legend()
plt.show()

# 3. 探究各种弱评估器的交叉验证结果
for booster in ["gbtree", "gblinear", "dart"]:
    reg = XGBR(n_estimators=180
               , learning_rate=0.1
               , random_state=420
               , booster=booster).fit(Xtrain, Ytrain)
    print(booster)
    print(reg.score(Xtest, Ytest))

# 4.探究gramm的效果大小
# ======【TIME WARNING: 1 min】=======#
axisx = np.arange(0, 5, 0.05)
rs = []
var = []
ge = []
for i in axisx:
    reg = XGBR(n_estimators=180, random_state=420, gamma=i)
    result = CVS(reg, Xtrain, Ytrain, cv=cv)
    rs.append(result.mean())
    var.append(result.var())
    ge.append((1 - result.mean()) ** 2 + result.var())
print(axisx[rs.index(max(rs))], max(rs), var[rs.index(max(rs))])
print(axisx[var.index(min(var))], rs[var.index(min(var))], min(var))
print(axisx[ge.index(min(ge))], rs[ge.index(min(ge))], var[ge.index(min(ge))], min(ge))
rs = np.array(rs)
var = np.array(var) * 0.1
plt.figure(figsize=(20, 5))
plt.plot(axisx, rs, c="black", label="XGB")
plt.plot(axisx, rs + var, c="red", linestyle='-.')
plt.plot(axisx, rs - var, c="red", linestyle='-.')
plt.legend()
plt.show()

# 如何控制过拟合
# 5. xgb实现法
import xgboost as xgb

dfull = xgb.DMatrix(X, y)

param1 = {'silent': True
    , 'obj': 'reg:linear'
    , "subsample": 1
    , "max_depth": 6
    , "eta": 0.3
    , "gamma": 0
    , "lambda": 1
    , "alpha": 0
    , "colsample_bytree": 1
    , "colsample_bylevel": 1
    , "colsample_bynode": 1
    , "nfold": 5}
num_round = 200

time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

fig, ax = plt.subplots(1, figsize=(15, 8))
ax.set_ylim(top=5)
ax.grid()
ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")
ax.legend(fontsize="xx-large")
plt.show()

param1 = {'silent': True
    , 'obj': 'reg:linear'
    , "subsample": 1
    , "max_depth": 6
    , "eta": 0.3
    , "gamma": 0
    , "lambda": 1
    , "alpha": 0
    , "colsample_bytree": 1
    , "colsample_bylevel": 1
    , "colsample_bynode": 1
    , "nfold": 5}
num_round = 200

time0 = time()
cvresult1 = xgb.cv(param1, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

fig, ax = plt.subplots(1, figsize=(15, 8))
ax.set_ylim(top=5)
ax.grid()
ax.plot(range(1, 201), cvresult1.iloc[:, 0], c="red", label="train,original")
ax.plot(range(1, 201), cvresult1.iloc[:, 2], c="orange", label="test,original")

param2 = {'silent': True
    , 'obj': 'reg:linear'
    , "max_depth": 2
    , "eta": 0.05
    , "gamma": 0
    , "lambda": 1
    , "alpha": 0
    , "colsample_bytree": 1
    , "colsample_bylevel": 0.4
    , "colsample_bynode": 1
    , "nfold": 5}

param3 = {'silent': True
    , 'obj': 'reg:linear'
    , "subsample": 1
    , "eta": 0.05
    , "gamma": 20
    , "lambda": 3.5
    , "alpha": 0.2
    , "max_depth": 4
    , "colsample_bytree": 0.4
    , "colsample_bylevel": 0.6
    , "colsample_bynode": 1
    , "nfold": 5}

time0 = time()
cvresult2 = xgb.cv(param2, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

time0 = time()
cvresult3 = xgb.cv(param3, dfull, num_round)
print(datetime.datetime.fromtimestamp(time() - time0).strftime("%M:%S:%f"))

ax.plot(range(1, 201), cvresult2.iloc[:, 0], c="green", label="train,last")
ax.plot(range(1, 201), cvresult2.iloc[:, 2], c="blue", label="test,last")
ax.plot(range(1, 201), cvresult3.iloc[:, 0], c="gray", label="train,this")
ax.plot(range(1, 201), cvresult3.iloc[:, 2], c="pink", label="test,this")
ax.legend(fontsize="xx-large")
plt.show()


