# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/4 16:19 
# License: bupt
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

data = pd.read_csv(r".\\day08_digit recognizor.csv")

X = data.iloc[:, 1:]
y = data.iloc[:, 0]
# print(X)
# print(X.shape)

# 1 将方差为0的数据进行清洗，进行删除
selector = VarianceThreshold()  # 实例化，不填参数默认方差为0
X_var0 = selector.fit_transform(X)  # 获取删除不合格特征之后的新特征矩阵

# 也可以直接写成 X = VairanceThreshold().fit_transform(X)

X_var0.shape  # (42000, 708)
pd.DataFrame(X_var0).head()

# 2 使用方差将其中一半的特征进行筛选过滤
import numpy as np

# X.var()#每一列的方差
X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
# print(X.var())
# print(X.var().values)
X.var().values

np.median(X.var().values)

print(X_fsvar.shape)  # (42000, 392)

# 3 KNN vs 随机森林在不同方差过滤效果下的对比
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score
import numpy as np

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)

# cross_val_score(KNN(), X, y, cv=5).mean()
#
# cross_val_score(KNN(), X_fsvar, y, cv=5).mean()
#
# cross_val_score(RFC(n_estimators=10, random_state=0), X, y, cv=5).mean()
#
# cross_val_score(RFC(n_estimators=10, random_state=0), X_fsvar, y, cv=5).mean()

#  4.相关性过滤 之卡方过滤
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# 假设在这里我一直我需要300个特征
X_fschi = SelectKBest(chi2, k=300).fit_transform(X_fsvar, y)
X_fschi.shape
# 模型结果下降，因为需要重新选择k值
# cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()
# 运用学习曲线，进行k的选择
import matplotlib.pyplot as plt

# score = []
# for i in range(390, 200, -10):
#     X_fschi = SelectKBest(chi2, k=i).fit_transform(X_fsvar, y)
#     once = cross_val_score(RFC(n_estimators=10, random_state=0), X_fschi, y, cv=5).mean()
#     score.append(once)
# plt.plot(range(390, 200, -10), score)
# plt.show()

# 时间太长，运行出来没有意义

#  4.2  相关性过滤之卡方过滤--利用显著性进行判断，
# k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
chivalue, pvalues_chi = chi2(X_fsvar, y)

print(chivalue.shape)

k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
print(k)
# X_fschi = SelectKBest(chi2, k=填写具体的k).fit_transform(X_fsvar, y)
# cross_val_score(RFC(n_estimators=10,random_state=0),X_fschi,y,cv=5).mean()


#  4.3 相关性过滤之卡方过滤--利用F检验进行判断，
# k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
from sklearn.feature_selection import f_classif

F, pvalues_f = f_classif(X_fsvar, y)

F

pvalues_f

k = F.shape[0] - (pvalues_f > 0.05).sum()
# print(k)
# X_fsF = SelectKBest(f_classif, k=填写具体的k).fit_transform(X_fsvar, y)
# cross_val_score(RFC(n_estimators=10,random_state=0),X_fsF,y,cv=5).mean()


#  4.4 相关性过滤之卡方过滤--利用互信息法进行判断，
# k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
from sklearn.feature_selection import mutual_info_classif as MIC

result = MIC(X_fsvar, y)

k = result.shape[0] - sum(result <= 0)

# X_fsmic = SelectKBest(MIC, k=填写具体的k).fit_transform(X_fsvar, y)
# cross_val_score(RFC(n_estimators=10,random_state=0),X_fsmic,y,cv=5).mean()

# 5 嵌入法实现对模型的特征选择，通常都是树结构

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier as RFC

RFC_ = RFC(n_estimators=10, random_state=0)

X_embedded = SelectFromModel(RFC_, threshold=0.005).fit_transform(X, y)

# 在这里我只想取出来有限的特征。0.005这个阈值对于有780个特征的数据来说，是非常高的阈值，因为平均每个特征
# 只能够分到大约0.001的feature_importances_

print(X_embedded.shape)

# 模型的维度明显被降低了
# 同样的，我们也可以画学习曲线来找最佳阈值

# ======【TIME WARNING：10 mins】======#

# import numpy as np
# import matplotlib.pyplot as plt
#
# RFC_.fit(X, y).feature_importances_
#
# threshold = np.linspace(0, (RFC_.fit(X, y).feature_importances_).max(), 20)
#
# score = []
# for i in threshold:
#     X_embedded = SelectFromModel(RFC_, threshold=i).fit_transform(X, y)
#     once = cross_val_score(RFC_, X_embedded, y, cv=5).mean()
#     score.append(once)
# plt.plot(threshold, score)
# plt.show()

X_embedded = SelectFromModel(RFC_, threshold=0.000564).fit_transform(X, y)
X_embedded.shape

cross_val_score(RFC_, X_embedded, y, cv=5).mean()

# =====【TIME WARNING：2 min】=====#
# 我们可能已经找到了现有模型下的最佳结果，如果我们调整一下随机森林的参数呢？
print(cross_val_score(RFC(n_estimators=100, random_state=0), X_embedded, y, cv=5).mean())



# 6 使用包装法对特征进行选择

from sklearn.feature_selection import RFE

RFC_ = RFC(n_estimators=10, random_state=0)
selector = RFE(RFC_, n_features_to_select=340, step=50).fit(X, y)

selector.support_.sum()  # 340

selector.ranking_

X_wrapper = selector.transform(X)

print(cross_val_score(RFC_, X_wrapper, y, cv=5).mean())

# score = []
# for i in range(1,751,50):
#     X_wrapper = RFE(RFC_,n_features_to_select=i, step=50).fit_transform(X,y)
#     once = cross_val_score(RFC_,X_wrapper,y,cv=5).mean()
#     score.append(once)
# plt.figure(figsize=[20,5])
# plt.plot(range(1,751,50),score)
# plt.xticks(range(1,751,50))
# plt.show()