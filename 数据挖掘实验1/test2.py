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
print(__doc__)

from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import *
from sklearn.feature_extraction.text import *
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA
from sklearn.preprocessing import *
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
np.random.seed(42)
categories = None
categories = [ 'alt.atheism','comp.graphics','comp.os.ms-windows.misc','comp.sys.ibm.pc.hardware']#选择几类的文件进行聚类操作，减少了数据量
dataset = fetch_20newsgroups(subset='all', categories= categories ,shuffle=True, random_state=42)
labels = dataset.target#labels_true
n_digits = np.unique(labels).shape[0]
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000,min_df=2, stop_words='english',use_idf=True)#向量化
data= vectorizer.fit_transform(dataset.data)
svd = TruncatedSVD(10)
normalizer = Normalizer(copy=False)
L =make_pipeline(svd, normalizer)
data = L.fit_transform(data)
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
# #############################################################################
# Visualize the results on PCA-reduced data  画图
#
# reduced_data = PCA(n_components=2).fit_transform(data)
# kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
# kmeans.fit(reduced_data)
#
# # Step size of the mesh. Decrease to increase the quality of the VQ.
# h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
#
# # Plot the decision boundary. For that, we will assign a color to each
# x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
# y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
#
# # Obtain labels for each point in mesh. Use last trained model.
# Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1)
# plt.clf()
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower')
# plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# # Plot the centroids as a white X
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 1],
#             marker='x', s=169, linewidths=3,
#             color='w', zorder=10)
# plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#           'Centroids are marked with white cross')
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)
# plt.xticks(())
# plt.yticks(())
# plt.show()
