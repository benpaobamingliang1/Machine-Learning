# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/3 21:26 
# License: bupt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import imblearn
data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
# 不太熟悉numpy的小伙伴，能够判断data的结构吗？
# 如果换成表是什么样子？


pd.DataFrame(data)

# 实现归一化
scaler = MinMaxScaler()  # 实例化
scaler = scaler.fit(data)  # fit，在这里本质是生成min(x)和max(x)
result = scaler.transform(data)  # 通过接口导出结果
# print(result)

result_ = scaler.fit_transform(data)  # 训练和导出结果一步达成

result1 = scaler.inverse_transform(result)  # 将归一化后的结果逆转
# print(result1)

# 使用MinMaxScaler的参数feature_range实现将数据归一化到[0,1]以外的范围中

x = np.array([[-1, 2], [-0.5, 6], [0, 10], [1, 18]])

# print(x)
scaler = MinMaxScaler(feature_range=[5, 10])  # 依然实例化
result = scaler.fit_transform(data)  # fit_transform一步导出结果
# print(result)
# 当X中的特征数量非常多的时候，fit会报错并表示，数据量太大了我计算不了
# 此时使用partial_fit作为训练接口
# scaler = scaler.partial_fit(data)

from sklearn.preprocessing import StandardScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

scaler = StandardScaler()  # 实例化
scaler.fit(data)  # fit，本质是生成均值和方差

print(scaler.mean_)
print(scaler.var_)

x_std = scaler.transform(data)  # 通过接口导出结果

# 导出的结果是一个数组，用mean()查看均值
print(x_std.mean())

x_std.std()  # 用std()查看方差

scaler.fit_transform(data)  # 使用fit_transform(data)一步达成结果

scaler.inverse_transform(x_std)  # 使用inverse_transform逆转标准化

import pandas as pd

data = pd.read_csv(r".\\day08_Narrativedata.csv"
                   , index_col=0
                   )  # index_col=0将第0列作为索引，不写则认为第0列为特征

# print(data.head())
# print(data.info())


# 2 对缺失值进行处理,直接使用pandas进行处理
import pandas as pd

data_ = pd.read_csv(r".\\day08_Narrativedata.csv"
                    , index_col=0
                    )  # index_col=0将第0列作为索引，不写则认为第0列为特征

data_.loc[:, "Age"] = data_.loc[:, "Age"].fillna(data_.loc[:, "Age"].mean())
# .fillna 在DataFrame里面直接进行填补
# print(data_.loc[:, "Age"])
data_.dropna(axis=0, inplace=True)

# data_.to_csv(r".\\day08_Narrativedata.csv", sep='\t', index=False, mode= 'w',
#          encoding='utf-8', header=0)
print(data_.info())
# .dropna(axis=0)删除所有有缺失值的行，.dropna(axis=1)删除所有有缺失值的列
# 参数inplace，为True表示在原数据集上进行修改，为False表示生成一个复制对象，不修改原数据，默认False


from sklearn.preprocessing import LabelEncoder

y = data_.iloc[:, -1]  # 要输入的是标签，不是特征矩阵，所以允许一维
#  print(y)
le = LabelEncoder()  # 实例化
le = le.fit(y)  # 导入数据
label = le.transform(y)  # transform接口调取结果

# 属性.classes_查看标签中究竟有多少类别
# print(le.classes_)
# 查看获取的结果label
# print(label)
le.fit_transform(y)  # 也可以直接fit_transform一步到位

le.inverse_transform(label)  # 使用inverse_transform可以逆转

# 2.2 preprocessing.OrdinalEncoder：特征专用，能够将分类特征转换为分类数值
from sklearn.preprocessing import OrdinalEncoder

# 接口categories_对应LabelEncoder的接口classes_，一模一样的功能
# data_ = data.copy()
# data_.head()
OrdinalEncoder().fit(data_.iloc[:, 1:-1]).categories_
data_.iloc[:, 1:-1] = OrdinalEncoder().fit_transform(data_.iloc[:, 1:-1])
print(data_.head())

# 2.3 这样的变化，让算法能够彻底领悟，原来三个取值是没有可计算性质的，是“有你就没有我”的不等概念。在我们的
# 数据中，性别和舱门，都是这样的名义变量。因此我们需要使用独热编码，将两个特征都转换为哑变量
# 总共有三种文字类型的关系变量，第一种是名义变量（没有任何关系），第二种是有序变量。第三种是有距变量
data_.head()

from sklearn.preprocessing import OneHotEncoder

X = data_.iloc[:, 1:-1]
print(X)
enc = OneHotEncoder(categories='auto').fit(X)
result = enc.transform(X).toarray()

# 依然可以直接一步到位，但为了给大家展示模型属性，所以还是写成了三步
result_ = OneHotEncoder(categories='auto').fit_transform(X).toarray()
# print(result_)
# 依然可以还原
pd.DataFrame(enc.inverse_transform(result))
# print(enc.get_feature_names())

# axis=1,表示跨行进行合并，也就是将两表左右相连，如果是axis=0，就是将量表上下相连
print(data_.shape)
print(result_.shape)
newdata = pd.concat([data_, pd.DataFrame(result)], axis=1)

newdata.drop(["Sex", "Embarked"], axis=1, inplace=True)
newdata.dropna(axis=0, inplace=True)
print(newdata.shape)
newdata.columns = ["Age", "Survived", "Female", "Male", "Embarked_C", "Embarked_Q", "Embarked_S"]

# print(newdata)

# 2.4 将年龄二值化

data_2 = data_.copy()

from sklearn.preprocessing import Binarizer

X = data_2.iloc[:, 0].values.reshape(-1, 1)  # 类为特征专用，所以不能使用一维数组
transformer = Binarizer(threshold=30).fit_transform(X)

data_2.iloc[:, 0] = transformer
# print(data_2)


# 3.1 去除重复值
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR

# 其实日常在导库的时候，并不是一次性能够知道我们要用的所有库的。通常都是在建模过程中逐渐导入需要的库

data_3 = pd.read_csv(r".\\day08_rankingcard.csv", index_col=0)
# 观察数据类型

# 去除重复值
data_3.drop_duplicates(inplace=True)  # inplace=True表示替换原数据

# 删除之后千万不要忘记，恢复索引
data_3.index = range(data_3.shape[0])

# print(data_3.info)

data_3["NumberOfDependents"].fillna(int(data_3["NumberOfDependents"].mean()), inplace=True)
# 这里用均值填补家庭人数这一项
# 如果你选择的是删除那些缺失了2.5%的特征，千万记得恢复索引哟~

print(data.info())
data_3.isnull().sum() / data_3.shape[0]


def fill_missing_rf(X, y, to_fill):
    """
    使用随机森林填补一个特征的缺失值的函数

    参数：
    X：要填补的特征矩阵
    y：完整的，没有缺失值的标签
    to_fill：字符串，要填补的那一列的名称
    """

    # 构建我们的新特征矩阵和新标签
    df = X.copy()
    fill = df.loc[:, to_fill]
    df = pd.concat([df.loc[:, df.columns != to_fill], pd.DataFrame(y)], axis=1)

    # 找出我们的训练集和测试集
    Ytrain = fill[fill.notnull()]
    Ytest = fill[fill.isnull()]
    Xtrain = df.iloc[Ytrain.index, :]
    Xtest = df.iloc[Ytest.index, :]

    # 用随机森林回归来填补缺失值
    from sklearn.ensemble import RandomForestRegressor as rfr
    rfr = rfr(n_estimators=100)
    rfr = rfr.fit(Xtrain, Ytrain)
    Ypredict = rfr.predict(Xtest)

    return Ypredict


X = data_3.iloc[:, 1:]
y = data_3["SeriousDlqin2yrs"]  # y = data.iloc[:,0]
X.shape  # (149391, 10)

# =====[TIME WARNING:1 min]=====#
y_pred = fill_missing_rf(X, y, "MonthlyIncome")

print(y_pred)
print(y_pred.shape)
print(type(y_pred))
# 注意可以通过以下代码检验数据是否数量相同
print(data_3.loc[:, "MonthlyIncome"].isnull().shape)
print(data_3.loc[data_3.loc[:, "MonthlyIncome"].isnull(), "MonthlyIncome"])
print(data_3.loc[data_3.loc[:, "MonthlyIncome"].isnull(), "MonthlyIncome"].shape)

y_pred.shape == data_3.loc[data_3.loc[:, "MonthlyIncome"].isnull(), "MonthlyIncome"].shape


# 确认我们的结果合理之后，我们就可以将数据覆盖了
data_3.loc[data_3.loc[:, "MonthlyIncome"].isnull(), "MonthlyIncome"] = y_pred

data_3.info()

# 3.2 去除异常值,后续在补充





# 3.3 样本不均衡问题
# 探索标签的分布
X = data_3.iloc[:, 1:]
y = data_3.iloc[:, 0]

y.value_counts()  # 查看每一类别值得数据量，查看样本是否均衡

n_sample = X.shape[0]

n_1_sample = y.value_counts()[1]
n_0_sample = y.value_counts()[0]

print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample, n_1_sample / n_sample, n_0_sample / n_sample))
# 样本个数：149165; 1占6.62%; 0占93.38%

# imblearn是专门用来处理不平衡数据集的库，在处理样本不均衡问题中性能高过sklearn很多
# imblearn里面也是一个个的类，也需要进行实例化，fit拟合，和sklearn用法相似

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)  # 实例化
X, y = sm.fit_sample(X, y)

n_sample_ = X.shape[0]  # 278584

pd.Series(y).value_counts()

n_1_sample = pd.Series(y).value_counts()[1]
n_0_sample = pd.Series(y).value_counts()[0]

print('样本个数：{}; 1占{:.2%}; 0占{:.2%}'.format(n_sample_, n_1_sample / n_sample_, n_0_sample / n_sample_))
# 样本个数：278584; 1占50.00%; 0占50.00%

# 3.4 分箱算法




