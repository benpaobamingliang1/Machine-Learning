# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/3 15:12 
# License: bupt
# 1.导包
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import numpy as np
from sklearn.metrics import brier_score_loss

# 2. 建立数据集
class_1 = 500
class_2 = 500  # 两个类别分别设定500个样本
centers = [[0.0, 0.0], [2.0, 2.0]]  # 设定两个类别的中心
clusters_std = [0.5, 0.5]  # 设定两个类别的方差
X, y = make_blobs(n_samples=[class_1, class_2],
                  centers=centers,
                  cluster_std=clusters_std,
                  random_state=0, shuffle=False)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y
                                                , test_size=0.3
                                                , random_state=420)

# print(Xtrain, Ytest)

# 3. 归一化，确保输入的矩阵不带有负数
# 先归一化，保证输入多项式朴素贝叶斯的特征矩阵中不带有负数
mms = MinMaxScaler().fit(Xtrain)
# print(Xtrain)
Xtrain_ = mms.transform(Xtrain)
# print(Xtrain_)
Xtest_ = mms.transform(Xtest)

# 4. 建立一个多项式朴素贝叶斯分类器
mnb = MultinomialNB().fit(Xtrain_, Ytrain)
# 重要属性：调用根据数据获取的，每个标签类的对数先验概率log(P(Y))
# 由于概率永远是在[0,1]之间，因此对数先验概率返回的永远是负值
print(mnb.class_log_prior_)
print(Ytrain)
Ytrain = np.unique(Ytrain)
print(Ytrain)
(Ytrain == 1).sum() / Ytrain.shape[0]
mnb.class_log_prior_.shape
# 可以使用np.exp来查看真正的概率值
print(np.exp(mnb.class_log_prior_))
# 重要属性：返回一个固定标签类别下的每个特征的对数概率log(P(Xi|y))
print(mnb.feature_log_prob_)
print(mnb.feature_log_prob_.shape)
# 重要属性：在fit时每个标签类别下包含的样本数。当fit接口中的sample_weight被设置时，
# 该接口返回的值也会受到加权的影响
mnb.class_count_
mnb.class_count_.shape

# 5.一些传统的接口
mnb.predict(Xtest_)
mnb.predict_proba(Xtest_)
mnb.score(Xtest_, Ytest)
print(brier_score_loss(Ytest, mnb.predict_proba(Xtest_)[:, 1], pos_label=1))

# 6. 不经过归一化处理
# 来试试看把Xtiain转换成分类型数据吧
# 注意我们的Xtrain没有经过归一化，因为做哑变量之后自然所有的数据就不会又负数了
from sklearn.preprocessing import KBinsDiscretizer

kbs = KBinsDiscretizer(n_bins=10, encode='onehot').fit(Xtrain)
Xtrain__ = kbs.transform(Xtrain)
Xtest__ = kbs.transform(Xtest)
mnb = MultinomialNB().fit(Xtrain__, Ytrain)
mnb.score(Xtest__, Ytest)
print(brier_score_loss(Ytest, mnb.predict_proba(Xtest__)[:, 1], pos_label=1))


from sklearn.naive_bayes import BernoulliNB
#普通来说我们应该使用二值化的类sklearn.preprocessing.Binarizer来将特征一个个二值化
#然而这样效率过低，因此我们选择归一化之后直接设置一个阈值
mms = MinMaxScaler().fit(Xtrain)
Xtrain_ = mms.transform(Xtrain)
Xtest_ = mms.transform(Xtest)
#不设置二值化
bnl_ = BernoulliNB().fit(Xtrain_, Ytrain)
bnl_.score(Xtest_,Ytest)
brier_score_loss(Ytest,bnl_.predict_proba(Xtest_)[:,1],pos_label=1)
#设置二值化阈值为0.5
bnl = BernoulliNB(binarize=0.5).fit(Xtrain_, Ytrain)
bnl.score(Xtest_,Ytest)
print(brier_score_loss(Ytest, bnl.predict_proba(Xtest_)[:, 1], pos_label=1))


