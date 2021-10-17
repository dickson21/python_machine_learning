"""
Subject : Principal Component Analysis (PCA)
Created : 2020-08-27
Author : dickson.cheng
Status : OK
Detail :
Principal Component Analysis (PCA)
PCA, generally called data reduction technique, 
is very useful feature selection technique as it uses linear algebra to transform the dataset into a compressed form. 
We can implement PCA feature selection technique with the help of PCA class of scikit-learn Python library. 
We can select number of principal components in the output
"""

from pandas import read_csv
from sklearn.decomposition import PCA
#path = r'C:\pima-indians-diabetes.csv'
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

pca = PCA(n_components = 3)
fit = pca.fit(X)
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)

"""
Explained Variance: [0.89 0.06 0.03]
[[-2.02e-03  9.78e-02  1.61e-02  6.08e-02  9.93e-01  1.40e-02  5.37e-04
  -3.56e-03]
 [-2.26e-02 -9.72e-01 -1.42e-01  5.79e-02  9.46e-02 -4.70e-02 -8.17e-04
  -1.40e-01]
 [-2.25e-02  1.43e-01 -9.22e-01 -3.07e-01  2.10e-02 -1.32e-01 -6.40e-04
  -1.25e-01]]
"""

