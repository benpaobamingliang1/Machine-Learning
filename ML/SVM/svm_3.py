# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/6 19:42 
# License: bupt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

weather = pd.read_csv(r"D:\DeskTop\python学习\Python-Machine-Learning-Algorithm-master\ML\SVM\weatherAUS5000.csv",
                      index_col=0)
print(weather.info())

# 1.1 将特征矩阵X和标签Y分开
X = weather.iloc[:, :-1]
Y = weather.iloc[:, -1]

# 探索缺失值
# 缺失值所占总值的比例 isnull().sum(全部的True)/X.shape[0]#我们要有不同的缺失值填补策略
print(X.isnull().mean())

# 分训练集和测试集随机抽样
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=420)

# 恢复索引
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])
print(Xtrain)

# 1.2 将标签编码
from sklearn.preprocessing import LabelEncoder  # 标签专用，第三章讲过

encorder = LabelEncoder().fit(Ytrain)  # 允许一维数据的输入的
# 认得了：有两类，YES和NO，YES是1，NO是0
# 使用训练集进行训练，然后在训练集和测试集上分别进行transform
Ytrain = pd.DataFrame(encorder.transform(Ytrain))
Ytest = pd.DataFrame(encorder.transform(Ytest))
# 如果我们的测试集中，出现了训练集中没有出现过的标签类别
# 比如说，测试集中有YES, NO, UNKNOWN
# 而我们的训练集中只有YES和NO

Ytrain.to_csv(r"D:\DeskTop\python学习\Python-Machine-Learning-Algorithm-master\ML\SVM\weather_Ytrain.csv")

# 描述性统计
Xtrain.describe([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).T

print(Xtrain.iloc[:, 0].value_counts())

Xtrain.loc[Xtrain.iloc[:, 0] == "2015-08-24", :]

Xtrain.loc[Xtrain["Rainfall"] >= 1, "RainToday"] = "Yes"
Xtrain.loc[Xtrain["Rainfall"] < 1, "RainToday"] = "No"
Xtrain.loc[Xtrain["Rainfall"] == np.nan, "RainToday"] = np.nan

int(Xtrain.loc[0, "Date"].split("-")[1])  # 提取出月份

Xtrain["Date"] = Xtrain["Date"].apply(lambda x: int(x.split("-")[1]))
# apply是对dataframe上的某一列进行处理的一个函数
# lambda x匿名函数，请在dataframe上这一列中的每一行帮我执行冒号后的命令

Xtrain["Date"] = Xtrain["Date"].apply(lambda x: int(x.split("-")[1]))
# apply是对dataframe上的某一列进行处理的一个函数
# lambda x匿名函数，请在dataframe上这一列中的每一行帮我执行冒号后的命令


Xtest["Date"] = Xtest["Date"].apply(lambda x: int(x.split("-")[1]))
Xtest = Xtest.rename(columns={"Date": "Month"})

# 首先找出，分类型特征都有哪些
cate = Xtrain.columns[Xtrain.dtypes == "object"].tolist()

# 除了特征类型为"object"的特征们，还有虽然用数字表示，但是本质为分类型特征的云层遮蔽程度
cloud = ["Cloud9am", "Cloud3pm"]
cate = cate + cloud

# 对于分类型特征，我们使用众数来进行填补
from sklearn.impute import SimpleImputer  # 0.20, conda, pip

si = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
# 注意，我们使用训练集数据来训练我们的填补器，本质是在生成训练集中的众数
si.fit(Xtrain.loc[:, cate])
# 然后我们用训练集中的众数来同时填补训练集和测试集
Xtrain.loc[:, cate] = si.transform(Xtrain.loc[:, cate])
Xtest.loc[:, cate] = si.transform(Xtest.loc[:, cate])
# 查看分类型特征是否依然存在缺失值
Xtrain.loc[:, cate].isnull().mean()

# 将所有的分类型变量编码为数字，一个类别是一个数字
from sklearn.preprocessing import OrdinalEncoder  # 只允许二维以上的数据进行输入

oe = OrdinalEncoder()

# 用训练集的编码结果来编码训练和测试特征矩阵
# 在这里如果测试特征矩阵报错，就说明测试集中出现了训练集中从未见过的类别
Xtrain.loc[:, cate] = oe.transform(Xtrain.loc[:, cate])
Xtest.loc[:, cate] = oe.transform(Xtest.loc[:, cate])

col = Xtrain.columns.tolist()
for i in cate:
    col.remove(i)
# 实例化模型，填补策略为"mean"表示均值
impmean = SimpleImputer(missing_values=np.nan, strategy="mean")
# 用训练集来fit模型
impmean = impmean.fit(Xtrain.loc[:, col])
# 分别在训练集和测试集上进行均值填补
Xtrain.loc[:, col] = impmean.transform(Xtrain.loc[:, col])
Xtest.loc[:, col] = impmean.transform(Xtest.loc[:, col])

from sklearn.preprocessing import StandardScaler  # 数据转换为均值为0，方差为1的数据

# 标准化不改变数据的分布，不会把数据变成正态分布的

ss = StandardScaler()
ss = ss.fit(Xtrain.loc[:, col])
Xtrain.loc[:, col] = ss.transform(Xtrain.loc[:, col])
Xtest.loc[:, col] = ss.transform(Xtest.loc[:, col])

from time import time  # 随时监控我们的模型的运行时间
import datetime
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, recall_score

Ytrain = Ytrain.iloc[:, 0].ravel()
Ytest = Ytest.iloc[:, 0].ravel()
# 建模选择自然是我们的支持向量机SVC，首先用核函数的学习曲线来选择核函数
# 我们希望同时观察，精确性，recall以及AUC分数
times = time()  # 因为SVM是计算量很大的模型，所以我们需要时刻监控我们的模型运行时间

for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    clf = SVC(kernel=kernel
              , gamma="auto"
              , degree=1
              , cache_size=5000
              ).fit(Xtrain, Ytrain)
    result = clf.predict(Xtest)
    score = clf.score(Xtest, Ytest)
    recall = recall_score(Ytest, result)
    auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
    print("%s 's testing accuracy %f, recall is %f', auc is %f" % (kernel, score, recall, auc))
    print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))

times = time()
for kernel in ["linear", "poly", "rbf", "sigmoid"]:
    clf = SVC(kernel=kernel
              , gamma="auto"
              , degree=1
              , cache_size=5000
              , class_weight="balanced"
              ).fit(Xtrain, Ytrain)
    result = clf.predict(Xtest)
    score = clf.score(Xtest, Ytest)
    recall = recall_score(Ytest, result)
    auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
    print("%s 's testing accuracy %f, recall is %f', auc is %f" % (kernel, score, recall, auc))
    print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))

times = time()
clf = SVC(kernel="linear"
          , gamma="auto"
          , cache_size=5000
          , class_weight={1: 15}  # 注意，这里写的其实是，类别1：10，隐藏了类别0：1这个比例
          ).fit(Xtrain, Ytrain)
result = clf.predict(Xtest)
score = clf.score(Xtest, Ytest)
recall = recall_score(Ytest, result)
auc = roc_auc_score(Ytest, clf.decision_function(Xtest))
print("testing accuracy %f, recall is %f', auc is %f" % (score, recall, auc))
print(datetime.datetime.fromtimestamp(time() - times).strftime("%M:%S:%f"))

irange = np.linspace(0.01,0.05,10)
for i in irange:
    times = time()
    clf = SVC(kernel = "linear"
              ,gamma="auto"
              ,cache_size = 5000
              ,class_weight = {1:1+i}
             ).fit(Xtrain, Ytrain)
    result = clf.predict(Xtest)
    score = clf.score(Xtest,Ytest)
    recall = recall_score(Ytest, result)
    auc = roc_auc_score(Ytest,clf.decision_function(Xtest))
    print("under ratio 1:%f testing accuracy %f, recall is %f', auc is %f" %(1+i,score,recall,auc))
    print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))



# 画精准率和召回率
times = time()
clf = SVC(kernel = "linear",C=3.1663157894736838,cache_size = 5000
          ,class_weight = "balanced"
         ).fit(Xtrain, Ytrain)
result = clf.predict(Xtest)
score = clf.score(Xtest,Ytest)
recall = recall_score(Ytest, result)
auc = roc_auc_score(Ytest,clf.decision_function(Xtest))
print("testing accuracy %f,recall is %f', auc is %f" % (score,recall,auc))
print(datetime.datetime.fromtimestamp(time()-times).strftime("%M:%S:%f"))
from sklearn.metrics import roc_curve as ROC
import matplotlib.pyplot as plt

FPR, Recall, thresholds = ROC(Ytest,clf.decision_function(Xtest),pos_label=1)

area = roc_auc_score(Ytest,clf.decision_function(Xtest))
plt.figure()
plt.plot(FPR, Recall, color='red',
         label='ROC curve (area = %0.2f)' % area)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('Recall')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
