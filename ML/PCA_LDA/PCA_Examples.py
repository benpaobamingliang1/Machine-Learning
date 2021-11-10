# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/8 16:31 
# License: bupt

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv(r".\\day08_digit recognizor.csv")

X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# (42000, 784)
pca_line = PCA().fit(X)
plt.figure(figsize=[20, 5])
plt.plot(np.cumsum(pca_line.explained_variance_ratio_))
plt.xlabel("number of components after dimension reduction")
plt.ylabel("cumulative explained variance ratio")
plt.show()

# ======【TIME WARNING：2mins 30s】======#

# score = []
# for i in range(1, 101, 10):
#     X_dr = PCA(i).fit_transform(X)
#     once = cross_val_score(RFC(n_estimators=10, random_state=0)
#                            , X_dr, y, cv=5).mean()
#     score.append(once)
# plt.figure(figsize=[20, 5])
# plt.plot(range(1, 101, 10), score)
# plt.show()


scorel = []
for i in range(10, 25):
    X_dr = PCA(i).fit_transform(X)
    once = cross_val_score(RFC(n_estimators=10, random_state=0), X_dr, y, cv=5).mean()
    scorel.append(once)
plt.figure(figsize=[20, 5])
plt.plot(range(10, 25), scorel)
plt.show()
print(max(scorel), ([*range(10, 25)][scorel.index(max(scorel))]))
print(scorel.index(max(scorel)))

X_dr = PCA(22).fit_transform(X)
# ======【TIME WARNING:1mins 30s】======#
# cross_val_score(RFC(n_estimators=100, random_state=0), X_dr, y, cv=5).mean()  # 0.946524472295366


from sklearn.neighbors import KNeighborsClassifier as KNN

# KNN()的值不填写默认=5    0.9698566872605972
cross_val_score(KNN(), X_dr, y, cv=5).mean()

# ======【TIME WARNING: 】======#
score = []
for i in range(10):
    X_dr = PCA(22).fit_transform(X)
    once = cross_val_score(KNN(i + 1), X_dr, y, cv=5).mean()
    score.append(once)
plt.figure(figsize=[20, 5])
plt.plot(range(10), score)
plt.show()

print(max(score), (score.index(max(scorel))))
print(score.index(max(score)))