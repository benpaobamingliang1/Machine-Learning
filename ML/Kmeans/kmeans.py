# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2021/11/8 20:52 
# License: bupt

# 1. 导库
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 自己创建数据集
X, y = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)
fig, ax1 = plt.subplots(1)
ax1.scatter(X[:, 0], X[:, 1]  # .scatter散点图
            , marker='o'  # 点的形状
            , s=8  # 点的大小
            )
plt.show()

# 如果我们想要看见这个点的分布，怎么办？
color = ["red", "pink", "orange", "gray"]
fig, ax1 = plt.subplots(1)

for i in range(4):
    ax1.scatter(X[y == i, 0], X[y == i, 1]
                , marker='o'  # 点的形状
                , s=8  # 点的大小
                , c=color[i]
                )
plt.show()

from sklearn.cluster import KMeans

n_clusters = 3
cluster = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
# 重要属性Labels_，查看聚好的类别，每个样本所对应的类
y_pred = cluster.labels_

# KMeans因为并不需要建立模型或者预测结果，因此我们只需要fit就能够得到聚类结果了
# KMeans也有接口predict和fit_predict，表示学习数据X并对X的类进行预测
# 但所得到的结果和我们不调用predict，直接fit之后调用属性labels一模一伴
pre = cluster.fit_predict(X)
pre

pre == y_pred  # 全都是True

# 我们什么时候需要predict呢？当数据量太大的时候！
# 其实我们不必使用所有的数据来寻找质心，少量的数据就可以帮助我们确定质心了
# 当我们数据量非常大的时候，我们可以使用部分数据来帮助我们确认质心
# 剩下的数据的聚类结果，使用predict来调用
cluster_smallsub = KMeans(n_clusters=n_clusters, random_state=0).fit(X[:200])

y_pred_ = cluster_smallsub.predict(X)

y_pred_

y_pred == y_pred_  # 数据量非常大的时候，效果会好
# 但从运行得出这样的结果，肯定与直接fit全部数据会不一致。有时候，当我们不要求那么精确，或者我们的数据量实在太大，那我们可以使用这种方法，使用接口predict
# 如果数据量还行，不是特别大，直接使用fit之后调用属性.labels_提出来

# 重要属性cLuster_centers_，查看质心
centroid = cluster.cluster_centers_
print(centroid)

# 重要属性inertia_，查看总距离平方和
inertia = cluster.inertia_
inertia  # 1903.4503741659223

color = ["red", "pink", "orange", "gray"]

fig, ax1 = plt.subplots(1)

for i in range(n_clusters):
    ax1.scatter(X[y_pred == i, 0], X[y_pred == i, 1]
                , marker='o'  # 点的形状
                , s=8  # 点的大小
                , c=color[i]
                )

ax1.scatter(centroid[:, 0], centroid[:, 1]
            , marker="x"
            , s=15
            , c="black")
plt.show()

# 如果我们把猜测的羡数换成4，Inertia会怎么样？
n_clusters = 4
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
inertia_

n_clusters = 5
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
inertia_

n_clusters = 6
cluster_ = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
inertia_ = cluster_.inertia_
inertia_

from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from time import time

# time（）：记下每一次time（）这一行命令时的时间戳
# 时间戳是一行数字，用来记录此时此刻的时间
t0 = time()
silhouette_score(X, y_pred)
time() - t0  # 0.007976055145263672
# 时间戳可以通过datetime中的函数fromtimestamp转换成真正的时间格式
import datetime

datetime.datetime.fromtimestamp(t0).strftime("%Y-%m-%d %H:%M:%S")

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # colormap
import numpy as np
import pandas as pd

n_clusters = 4
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([-0.1, 1])
ax1.set_ylim([0, X.shape[0] + (n_clusters + 1) * 10])
clusterer = KMeans(n_clusters=n_clusters, random_state=10).fit(X)
cluster_labels = clusterer.labels_
silhouette_avg = silhouette_score(X, cluster_labels)
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)

sample_silhouette_values = silhouette_samples(X, cluster_labels)

y_lower = 10

for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)

    ax1.fill_betweenx(np.arange(y_lower, y_upper)
                      , ith_cluster_silhouette_values
                      , facecolor=color
                      , alpha=0.7
                      )

    ax1.text(-0.05
             , y_lower + 0.5 * size_cluster_i
             , str(i))

    y_lower = y_upper + 10

ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")

ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])

ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)

ax2.scatter(X[:, 0], X[:, 1]
            , marker='o'
            , s=8
            , c=colors
            )

centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1], marker='x',
            c="red", alpha=1, s=200)

ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")

plt.suptitle(("Silhouette analysis for KMeans clustering on sample data"
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
# 对两个序列中的点进行距离匹配的函数
from sklearn.datasets import load_sample_image
# 导入图片数据所用的类
from sklearn.utils import shuffle  # 洗牌


# 实例化，导入颐和园的图片
china = load_sample_image("china.jpg")

#查看数据类型
print(china.dtype)

# 图像可视化
plt.figure(figsize=(15,15))
plt.imshow(china) #导入3维数组形成的图片
plt.show()

n_clusters = 64

china = np.array(china, dtype=np.float64) / china.max()
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

#首先，先使用1000个数据来找出质心
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(image_array_sample)

#使用质心来替换所有的样本
image_kmeans = image_array.copy()

plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)
plt.show()


plt.figure(figsize=(10,10))
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(image_kmeans)
plt.show()
