# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/3 11:44 
# License: bupt

# 1.布里尔分数Brier Score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt
# 1.导入数据
from sklearn.naive_bayes import GaussianNB

digits = load_digits()
x, y = digits.data, digits.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=420)
# 注意，第一个参数是真实标签，第二个参数是预测出的概率值
# 在二分类情况下，接口predict_proba会返回两列，但SVC的接口decision_function却只会返回一列
# 要随时注意，使用了怎样的概率分类器，以辨别查找置信度的接口，以及这些接口的结构

# # 2.建模，探索结果
gnb = GaussianNB().fit(x_train, y_train)
# # 查看预测结果
y_pred = gnb.predict_proba(x_test)
# print(y_pred)
# #brier_score_loss(y_test, y_pred[:, 1], pos_label=1)
# # 我们的pos_label与prob中的索引一致，就可以查看这个类别下的布里尔分数是多少
# from sklearn.metrics import brier_score_loss
#
# #brier_score_loss(y_test, y_pred[:, 8], pos_label=8)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as LR

#
logi = LR(C=1., solver='lbfgs', max_iter=3000, multi_class="auto").fit(x_train, y_train)
svc = SVC(kernel="linear", gamma=1).fit(x_train, y_train)
# brier_score_loss(y_test, logi.predict_proba(x_test)[:, 1], pos_label=1)
# # 由于SVC的置信度并不是概率，为了可比性，我们需要将SVC的置信度“距离”归一化，压缩到[0,1]之间
svc_prob = (svc.decision_function(x_test) -
            svc.decision_function(x_test).min()) / (svc.decision_function(x_test).max() -
                                                    svc.decision_function(x_test).min())
# brier_score_loss(y_test, svc_prob[:, 1], pos_label=1)
# import pandas as pd
#
# name = ["Bayes", "Logistic", "SVC"]
# color = ["red", "black", "orange"]
# df = pd.DataFrame(index=range(10), columns=name)
# for i in range(10):
#     df.loc[i, name[0]] = brier_score_loss(y_test, y_pred[:, i], pos_label=i)
#     df.loc[i, name[1]] = brier_score_loss(y_test, logi.predict_proba(x_test)[:, i], pos_label=i)
#     df.loc[i, name[2]] = brier_score_loss(y_test, svc_prob[:, i], pos_label=i)
# for i in range(df.shape[1]):
#     plt.plot(range(10), df.iloc[:, i], c=color[i])
# plt.legend()
# plt.show()

from sklearn.metrics import log_loss

print(log_loss(y_test, y_pred))
log_loss(y_test, logi.predict_proba(x_test))
log_loss(y_test, svc_prob)
