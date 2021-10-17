"""
Subject : Hierarchical Clustering
Created : 2020-08-28
Author : dickson.cheng
Status : OK

"""

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from pandas import read_csv
#path = r"C:\pima-indians-diabetes.csv"
path=r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]
data.shape
#(768, 9)
data.head()

patient_data = data.iloc[:, 3:5].values
import scipy.cluster.hierarchy as shc
plt.figure(figsize = (10, 7))
plt.title("Patient Dendograms")
dend = shc.dendrogram(shc.linkage(data, method = 'ward'))

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
cluster.fit_predict(patient_data)
plt.figure(figsize = (10, 7))
plt.scatter(patient_data[:,0], patient_data[:,1], c = cluster.labels_, cmap = 'rainbow')
