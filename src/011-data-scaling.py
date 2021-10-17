"""
Subject : data scaling
Created : 2020-08-26
Author : dickson.cheng
Status : 
 NotOK - cmd console
 OK    - jupyter lab
Detail : 
Most probably our dataset comprises of the attributes with varying scale, 
but we cannot provide such data to ML algorithm hence it requires rescaling. 
Data rescaling makes sure that attributes are at same scale. 
Generally, attributes are rescaled into the range of 0 and 1. 
ML algorithms like gradient descent and k-Nearest Neighbors requires scaled data. 
We can rescale the data with the help of MinMaxScaler class of scikit-learn Python library.

Ref:
module list
...
scikit-image              0.15.0           py37ha925a31_0
scikit-learn              0.21.3           py37h6288b17_0
...

Scikit learn DLL load failed in anaconda
https://stackoverflow.com/questions/55201924/scikit-learn-dll-load-failed-in-anaconda

Python scipy module import error due to missing ._ufuncs dll
https://stackoverflow.com/questions/39020361/python-scipy-module-import-error-due-to-missing-ufuncs-dll

"""

from pandas import read_csv
from numpy import set_printoptions
from sklearn import preprocessing

#path = r'C:\pima-indians-diabetes.csv'
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values

data_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data_rescaled = data_scaler.fit_transform(array)

set_printoptions(precision=1)
print ("\nScaled data:\n", data_rescaled[0:10])


"""
Output
from sklearn import preprocessing : module not found

Console - NotOK
Traceback (most recent call last):
  File "c:\myprogram\python\machine_learning\008-data-scaling.py", line 20, in <module>
    from sklearn import preprocessing

Jupyterlab output - OK
Scaled data:
 [[0.4 0.7 0.6 0.4 0.  0.5 0.2 0.5 1. ]
 [0.1 0.4 0.5 0.3 0.  0.4 0.1 0.2 0. ]
 [0.5 0.9 0.5 0.  0.  0.3 0.3 0.2 1. ]
 [0.1 0.4 0.5 0.2 0.1 0.4 0.  0.  0. ]
 [0.  0.7 0.3 0.4 0.2 0.6 0.9 0.2 1. ]
 [0.3 0.6 0.6 0.  0.  0.4 0.1 0.2 0. ]
 [0.2 0.4 0.4 0.3 0.1 0.5 0.1 0.1 1. ]
 [0.6 0.6 0.  0.  0.  0.5 0.  0.1 0. ]
 [0.1 1.  0.6 0.5 0.6 0.5 0.  0.5 1. ]
 [0.5 0.6 0.8 0.  0.  0.  0.1 0.6 1. ]]


"""