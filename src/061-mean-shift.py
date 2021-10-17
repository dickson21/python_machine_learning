"""
Subject : Mean shift
Created : 2020-08-28
Author : dickson.cheng
Status : OK
Detail :
Introduction to Mean-Shift Algorithm
As discussed earlier, 
it is another powerful clustering algorithm used in unsupervised learning. Unlike K-means clustering, 
it does not make any assumptions; hence it is a non-parametric algorithm.

Mean-shift algorithm basically assigns the datapoints to the clusters iteratively by 
shifting points towards the highest density of datapoints i.e. cluster centroid.

The difference between K-Means algorithm and Mean-Shift is that 
later one does not need to specify the number of clusters in advance because 
the number of clusters will be determined by the algorithm w.r.t data.

Working of Mean-Shift Algorithm
We can understand the working of Mean-Shift clustering algorithm with the help of following steps −
Step 1 − First, start with the data points assigned to a cluster of their own.
Step 2 − Next, this algorithm will compute the centroids.
Step 3 − In this step, location of new centroids will be updated.
Step 4 − Now, the process will be iterated and moved to the higher density region.
Step 5 − At last, it will be stopped once the centroids reach at position from where it cannot move further.

Ref:
mean shift
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_clustering_algorithms_mean_shift.htm

"""

import numpy as np
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.datasets.samples_generator import make_blobs
centers = [[3,3,3],[4,5,5],[3,10,10]]
X, _ = make_blobs(n_samples = 700, centers = centers, cluster_std = 0.5)
plt.scatter(X[:,0],X[:,1])
plt.show()

ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_
print(cluster_centers)
n_clusters_ = len(np.unique(labels))
print("Estimated clusters:", n_clusters_)
colors = 10*['r.','g.','b.','c.','k.','y.','m.']

for i in range(len(X)):
   plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 3)
plt.scatter(cluster_centers[:,0],cluster_centers[:,1],
   marker = ".",color = 'k', s = 20, linewidths = 5, zorder = 10)
plt.show()


"""
[[ 2.98462798 9.9733794 10.02629344]
[ 3.94758484 4.99122771 4.99349433]
[ 3.00788996 3.03851268 2.99183033]]
Estimated clusters: 3

"""



