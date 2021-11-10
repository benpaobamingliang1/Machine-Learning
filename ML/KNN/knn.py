# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/1 11:15 
# License: bupt
import numpy  as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics

# 一 1.数据加载和数据预处理
iris = datasets.load_iris()
x = iris.data
y = iris.target
y = y.reshape(-1, 1)

df = pd.DataFrame(data=x, index=None, columns=iris.feature_names)
df['class'] = iris.target
df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
# print(df)


# 2.划分训练集和测试集
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3, random_state=50, stratify=y)

print(x_train.shape, y_train.shape)


# 二、 核心算法实现
# 2.1 定义距离函数
# a可以是矩阵或者向量，b只能是向量，print(l1_distance(x_train, x_test[0].reshape(1, -1)))
def l1_distance(a, b):
    return np.sum(np.abs(a - b), axis=1)


def l2_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2, axis=1))


# 2.2 分类器
class kNN(object):
    # 定义一个初始化方法，__init__是类的构造方法
    # 所有类方法的第一个参数就是类本身
    def __init__(self, n_neighbors=1, dist_func=l1_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func

    # 训练模型方法
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    # 模型预测方法 print(np.zeros((x_train.shape[0], 1)))
    def predict(self, x):
        # 初始化预测分类数组
        y_pred = np.zeros((x.shape[0], 1), dtype=self.y_train.dtype)
        # 遍历输入的x数据点，取出每一个数据点的序号i和数据x_test
        for i, x_test in enumerate(x):
            # 计算x_test和所有训练数据的距离
            distances = self.dist_func(self.x_train, x_test)

            # 得到的距离按照由近到远排序，取出索引值
            nn_index = np.argsort(distances)

            # 得到的距离按照由近到远排序， 取出索引值
            nn_y = self.y_train[nn_index[:self .n_neighbors]].ravel()

            # 统计出类别频率最高的那个，赋值给y_pred[i]
            y_pred[i] = np.argmax(np.bincount(nn_y))

        return y_pred


# 三、 测试
# 3.1 定义一个knn实例
knn = kNN(n_neighbors=3)
# 训练模型
knn.fit(x_train, y_train)
# 传入测试数据，做预测
y_pred = knn.predict(x_test)

print(y_pred)

# 3.3.1 求出预测准确率
accuracy = metrics.accuracy_score(y_test, y_pred)

print("预测准确率", accuracy)


# 3.3.3 定义一个knn实例
knn = kNN()
# 训练模型
knn.fit(x_train, y_train)
# 保存结果
result_list = []
# 针对不同的参数选择，做预测
for p in [1,2] :
    knn.dist_func = l1_distance if p == 1 else l2_distance
    # 传入测试数据， 做预测
    for k in range(1,10,2):
        knn.n_neighbors = k
        # 传入测试数据，做预测
        y_pred = knn.predict(x_test)
        # 求出准确预测率
        accuracy = metrics.accuracy_score(y_test, y_pred)
        result_list.append([k, 'l1_distance' if p == 1 else 'l2_distance', accuracy])

df = pd.DataFrame(result_list, columns=['k', '距离函数', '预测准确率'])

print(df)