"""
Subject : Hierarchical Clustering
Created : 2020-08-28
Author : dickson.cheng
Status : OK

Detail :
Introduction to Hierarchical Clustering
Hierarchical clustering is another unsupervised learning algorithm that is used to group together the unlabeled data points having similar characteristics. Hierarchical clustering algorithms falls into following two categories.

Agglomerative hierarchical algorithms − In agglomerative hierarchical algorithms, each data point is treated as a single cluster and then successively merge or agglomerate (bottom-up approach) the pairs of clusters. The hierarchy of the clusters is represented as a dendrogram or tree structure.

Divisive hierarchical algorithms − On the other hand, in divisive hierarchical algorithms, all the data points are treated as one big cluster and the process of clustering involves dividing (Top-down approach) the one big cluster into various small clusters.

Steps to Perform Agglomerative Hierarchical Clustering
We are going to explain the most used and important Hierarchical clustering i.e. agglomerative. 
The steps to perform the same is as follows −

Step 1 − 
Treat each data point as single cluster. Hence, we will be having, say K clusters at start. 
The number of data points will also be K at start.
Step 2 − 
Now, in this step we need to form a big cluster by joining two closet datapoints. 
This will result in total of K-1 clusters.
Step 3 − 
Now, to form more clusters we need to join two closet clusters. 
This will result in total of K-2 clusters.
Step 4 − 
Now, to form one big cluster repeat the above three steps until K would become 0 
i.e. no more data points left to join.
Step 5 − 
At last, after making one single big cluster, dendrograms will be used to divide into multiple clusters depending upon the problem.

Role of Dendrograms in Agglomerative Hierarchical Clustering
As we discussed in the last step, the role of dendrogram starts once the big cluster is formed. 
Dendrogram will be used to split the clusters into multiple cluster of related data points depending upon our problem. 
It can be understood with the help of following example −

Ref :
Hierarchical Clustering
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_clustering_algorithms_hierarchical.htm

"""

import matplotlib.pyplot as plt
import numpy as np

X = np.array(
   [[7,8],[12,20],[17,19],[26,15],[32,37],[87,75],[73,85], [62,80],[73,60],[87,96],])
labels = range(1, 11)
plt.figure(figsize = (10, 7))
plt.subplots_adjust(bottom = 0.1)
plt.scatter(X[:,0],X[:,1], label = 'True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
   plt.annotate(
      label,xy = (x, y), xytext = (-3, 3),textcoords = 'offset points', ha = 'right', va = 'bottom')
plt.show()

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
linked = linkage(X, 'single')
labelList = range(1, 11)
plt.figure(figsize = (10, 7))
dendrogram(linked, orientation = 'top',labels = labelList, 
   distance_sort ='descending',show_leaf_counts = True)
plt.show()

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(X)

plt.scatter(X[:,0],X[:,1], c = cluster.labels_, cmap = 'rainbow')


