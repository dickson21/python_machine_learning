"""
Subject : K Means
Created : 2020-08-28
Author : dickson.cheng
Status : OK

Detail :
Introduction to Clustering
Clustering methods are one of the most useful unsupervised ML methods. These methods are used to find similarity as well as the relationship patterns among data samples and then cluster those samples into groups having similarity based on features.

Clustering is important because it determines the intrinsic grouping among the present unlabeled data. They basically make some assumptions about data points to constitute their similarity. Each assumption will construct different but equally valid clusters.

For example, below is the diagram which shows clustering system grouped together the similar kind of data in different clusters −

Cluster Formation Methods
It is not necessary that clusters will be formed in spherical form. Followings are some other cluster formation methods −

Density-based
In these methods, the clusters are formed as the dense region. The advantage of these methods is that they have good accuracy as well as good ability to merge two clusters. Ex. Density-Based Spatial Clustering of Applications with Noise (DBSCAN), Ordering Points to identify Clustering structure (OPTICS) etc.

Hierarchical-based
In these methods, the clusters are formed as a tree type structure based on the hierarchy. They have two categories namely, Agglomerative (Bottom up approach) and Divisive (Top down approach). Ex. Clustering using Representatives (CURE), Balanced iterative Reducing Clustering using Hierarchies (BIRCH) etc.

Partitioning
In these methods, the clusters are formed by portioning the objects into k clusters. Number of clusters will be equal to the number of partitions. Ex. K-means, Clustering Large Applications based upon randomized Search (CLARANS).

Grid
In these methods, the clusters are formed as a grid like structure. The advantage of these methods is that all the clustering operation done on these grids are fast and independent of the number of data objects. Ex. Statistical Information Grid (STING), Clustering in Quest (CLIQUE).

Measuring Clustering Performance
One of the most important consideration regarding ML model is assessing its performance or you can say model’s quality. In case of supervised learning algorithms, assessing the quality of our model is easy because we already have labels for every example.

On the other hand, in case of unsupervised learning algorithms we are not that much blessed because we deal with unlabeled data. But still we have some metrics that give the practitioner an insight about the happening of change in clusters depending on algorithm.

Before we deep dive into such metrics, we must understand that these metrics only evaluates the comparative performance of models against each other rather than measuring the validity of the model’s prediction. Followings are some of the metrics that we can deploy on clustering algorithms to measure the quality of model −

Silhouette Analysis
Silhouette analysis used to check the quality of clustering model by measuring the distance between the clusters. It basically provides us a way to assess the parameters like number of clusters with the help of Silhouette score. This score measures how close each point in one cluster is to points in the neighboring clusters.

Analysis of Silhouette Score
Analysis of Silhouette Score − The range of Silhouette score is [-1, 1].

Types of ML Clustering Algorithms
The following are the most important and useful ML clustering algorithms −

K-means Clustering
This clustering algorithm computes the centroids and iterates until we it finds optimal centroid. It assumes that the number of clusters are already known. It is also called flat clustering algorithm. The number of clusters identified from data by algorithm is represented by ‘K’ in K-means.

Mean-Shift Algorithm
It is another powerful clustering algorithm used in unsupervised learning. Unlike K-means clustering, it does not make any assumptions hence it is a non-parametric algorithm.

Hierarchical Clustering
It is another unsupervised learning algorithm that is used to group together the unlabeled data points having similar characteristics.

We will be discussing all these algorithms in detail in the upcoming chapters.

Applications of Clustering
We can find clustering useful in the following areas −
01# Data summarization and compression − 
Clustering is widely used in the areas where we require data summarization, compression and reduction as well. 
The examples are image processing and vector quantization.

02# Collaborative systems and customer segmentation − 
Since clustering can be used to find similar products or same kind of users, 
it can be used in the area of collaborative systems and customer segmentation.

03# Serve as a key intermediate step for other data mining tasks − 
Cluster analysis can generate a compact summary of data for 
classification, testing, hypothesis generation; 
hence, it serves as a key intermediate step for other data mining tasks also.

04# Trend detection in dynamic data − 
Clustering can also be used for trend detection in dynamic data by making various clusters of similar trends.

05# ocial network analysis − 
Clustering can be used in social network analysis. 
The examples are generating sequences in images, videos or audios.

06# Biological data analysis − 
Clustering can also be used to make clusters of images, 
videos hence it can successfully be used in biological data analysis.


Introduction to K-Means Algorithm
K-means clustering algorithm computes the centroids and 
iterates until we it finds optimal centroid. It assumes that the number of clusters are already known. 
It is also called flat clustering algorithm. 
The number of clusters identified from data by algorithm is represented by ‘K’ in K-means.

In this algorithm, the data points are assigned to a cluster in such a manner that the sum of the squared distance 
between the data points and centroid would be minimum. 
It is to be understood that less variation within the clusters will lead to more similar data points within same cluster.

Working of K-Means Algorithm
We can understand the working of K-Means clustering algorithm with the help of following steps −
Step 1 − First, we need to specify the number of clusters, K, need to be generated by this algorithm.
Step 2 − Next, randomly select K data points and assign each data point to a cluster. In simple words, classify the data based on the number of data points.
Step 3 − Now it will compute the cluster centroids.
Step 4 − Next, keep iterating the following until we find optimal centroid which is the assignment of data points to the clusters that are not changing any more

4.1 − First, the sum of squared distance between data points and centroids would be computed.
4.2 − Now, we have to assign each data point to the cluster that is closer than other cluster (centroid).
4.3 − At last compute the centroids for the clusters by taking the average of all data points of that cluster.

K-means follows Expectation-Maximization approach to solve the problem. 
The Expectation-step is used for assigning the data points to the closest cluster and 
the Maximization-step is used for computing the centroid of each cluster.

While working with K-means algorithm we need to take care of the following things −

While working with clustering algorithms including K-Means, 
it is recommended to standardize the data because 
such algorithms use distance-based measurement to determine the similarity between data points.

Due to the iterative nature of K-Means and random initialization of centroids, 
K-Means may stick in a local optimum and may not converge to global optimum. 
That is why it is recommended to use different initializations of centroids.


Applications of K-Means Clustering Algorithm
The main goals of cluster analysis are −
To get a meaningful intuition from the data we are working with.
Cluster-then-predict where different models will be built for different subgroups.
To fulfill the above-mentioned goals, K-means clustering is performing well enough. It can be used in following applications −
01# Market segmentation
02# Document Clustering
03# Image segmentation
04# Image compression
05# Customer segmentation
06# Analyzing the trend on dynamic data

Ref :
Clustering
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_clustering_algorithms_overview.htm

"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.cluster import KMeans

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples = 400, centers = 4, cluster_std = 0.60, random_state = 0)

plt.scatter(X[:, 0], X[:, 1], s = 20);
plt.show()

kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples = 400, centers = 4, cluster_std = 0.60, random_state = 0)

plt.scatter(X[:, 0], X[:, 1], c = y_kmeans, s = 20, cmap = 'summer')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c = 'blue', s = 100, alpha = 0.9);
plt.show()

