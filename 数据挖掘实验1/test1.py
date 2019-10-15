"""
===========================================================
A demo of K-Means clustering on the handwritten digits data
===========================================================

In this example we compare the various initialization strategies for
K-means in terms of runtime and quality of the results.

As the ground truth is known here, we also apply different cluster
quality metrics to judge the goodness of fit of the cluster labels to the
ground truth.

Cluster quality metrics evaluated (see :ref:`clustering_evaluation` for
definitions and discussions of the metrics):

=========== ========================================================
Shorthand    full name
=========== ========================================================
homo         homogeneity score
compl        completeness score
v-meas       V measure
ARI          adjusted Rand index
AMI          adjusted mutual information
silhouette   silhouette coefficient
=========== ========================================================

"""
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
digits = load_digits()                #数据集
data = scale(digits.data)            #标准化后的每个元素为8*8矩阵的数组
n_samples, n_features = data.shape   #(179 64)  数据总数 特征总数
labels = digits.target               #数组 每一个特征对应的真实的分组的标签
n_digits = len(np.unique(labels))    # np.unique()对数组进行去重 分的组数 总的类数
sample_size = 300
print(82 * '_')
print('init\t\ttime\thomo\tcompl\tNMI') #每一列 属性
def cluster_test(estimator, name, data):
    t0 = time()         #开始事件
    estimator.fit(data) #根据聚类器 开始聚类
    if hasattr(estimator, 'labels_'):#GaussionMixture没有label_属性
        pre_labels = estimator.labels_
    else:
        pre_labels = estimator.predict(data)
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f'
      % (name, (time() - t0),               #聚类时间
             metrics.homogeneity_score(labels, pre_labels),#每一个聚出的类仅包含一个类别的程度度量 越大越好
             metrics.completeness_score(labels, pre_labels),#每一个类别被指向相同聚出的类的程度度量
             metrics.adjusted_mutual_info_score(labels, pre_labels,average_method='arithmetic'),
             ))

cluster_test(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),name="KM_km++", data=data)
cluster_test(KMeans(init='random', n_clusters=n_digits, n_init=10),name="KM_random", data=data)
pca = PCA(n_components=n_digits).fit(data)   #pca先找出簇的中心点
cluster_test(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),name="KM_pca",data=data)
#cluster_test(AffinityPropagation(damping=0.6,max_iter=100,preference=-50),name="AP",data=data)#阻尼系数和最大迭代次数
bandwidth=estimate_bandwidth(data, quantile=0.2, n_samples=900)   #本身不知道要聚类的数目 所以做出来的聚类效果并不是很好
cluster_test(MeanShift(bandwidth=bandwidth, bin_seeding=True),name="Meshift",data=data)
cluster_test(SpectralClustering(n_clusters=n_digits, eigen_solver='arpack',affinity="nearest_neighbors"),name="SCluster", data=data)
cluster_test(AffinityPropagation(max_iter=100,convergence_iter=10,copy=False),name="ACluster", data=data)
cluster_test(GaussianMixture(n_components=n_digits),name='GMixture', data=data)
cluster_test(DBSCAN(),name="DBScan", data=data)
print(82 * '_')