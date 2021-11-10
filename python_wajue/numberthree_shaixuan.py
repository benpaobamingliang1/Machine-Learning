import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
#  通过随机森林分离变量对辛烷值含量的重要性
from time import time

path = r'Molecular_Descriptor.xlsx'
data = pd.read_excel(path, header=0)
print(data)
PON = data.iloc[:, -1].values
print(PON)
factors = data.iloc[:, 0:729].values
print(factors)
print(PON)
forest = RandomForestRegressor(n_estimators=500, random_state=0, max_depth=100, n_jobs=2)
x_train, x_test, y_train, y_test = train_test_split(factors, PON, test_size=0.3,shuffle=True, random_state=0)
forest.fit(x_train, y_train)
feat_labels = data.columns
print(feat_labels)
importances = forest.feature_importances_
print(importances)
indices = np.argsort(importances)[::-1]
print(x_train.shape[1])
print(x_train.shape[0])
m = []
for f in range(20):
    d = data[feat_labels[indices[f]]]
    m.append(d)
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

m.append(PON)
data1 = pd.DataFrame(m).T
data1.to_excel('data2.xlsx')
data1.to_csv('data2.csv')

# from sklearn.ensemble import GradientBoostingClassifier
# gbdt = GradientBoostingClassifier()
# gbdt.fit(training_data, training_labels)  # 训练。喝杯咖啡吧
# GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
#               max_depth=3, max_features=None, max_leaf_nodes=None,
#               min_samples_leaf=1, min_samples_split=2,
#               min_weight_fraction_leaf=0.0, n_estimators=100,
#               random_state=None, subsample=1.0, verbose=0,
#               warm_start=False)
# gbdt.feature_importances_   # 据此选取重要的特征
# gbdt.feature_importances_.shape
# (19630,)


