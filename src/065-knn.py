"""
Subject : KNN Algorithm - Finding Nearest Neighbors (KNN-regressor)
Created : 2020-08-28
Author : dickson.cheng
Status : NotOK


Ref :
KNN Algorithm - Finding Nearest Neighbors
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_knn_algorithm_finding_nearest_neighbors.htm
"""

import numpy as np
import pandas as pd
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
headernames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
data = pd.read_csv(path, names = headernames)
array = data.values
X = array[:,:2]
Y = array[:,2]
data.shape
# output:(150, 5)
"""
-- NotOK --
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors = 10)
knnr.fit(X, y)
print ("The MSE is:",format(np.power(y-knnr.predict(X),2).mean()))
"""

X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsRegressor
knnr = KNeighborsRegressor(n_neighbors = 3)
knnr.fit(X, y)
print(knnr.predict([[2.5]]))


"""
[0.66666667]
"""


