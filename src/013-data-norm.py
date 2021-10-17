"""
Subject : data normalization
Created : 2020-08-27
Author : dickson.cheng

Detail :
Standardization
Another useful data preprocessing technique which is basically used to transform the data attributes with a Gaussian distribution. 
It differs the mean and SD (Standard Deviation) to a standard Gaussian distribution with a mean of 0 and a SD of 1. 
This technique is useful in ML algorithms like linear regression, 
logistic regression that assumes a Gaussian distribution in input dataset and produce better results with rescaled data. 
We can standardize the data (mean = 0 and SD =1) with the help of StandardScaler class of scikit-learn Python library.
"""

from sklearn.preprocessing import StandardScaler
from pandas import read_csv
from numpy import set_printoptions
#path = r'C:\pima-indians-diabetes.csv'
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values
data_scaler = StandardScaler().fit(array)
data_rescaled = data_scaler.transform(array)
set_printoptions(precision=2)
print ("\nRescaled data:\n", data_rescaled [0:5])

"""
Rescaled data:
 [[ 0.64  0.85  0.15  0.91 -0.69  0.2   0.47  1.43  1.37]
 [-0.84 -1.12 -0.16  0.53 -0.69 -0.68 -0.37 -0.19 -0.73]
 [ 1.23  1.94 -0.26 -1.29 -0.69 -1.1   0.6  -0.11  1.37]
 [-0.84 -1.   -0.16  0.15  0.12 -0.49 -0.92 -1.04 -0.73]
 [-1.14  0.5  -1.5   0.91  0.77  1.41  5.48 -0.02  1.37]]
"""

