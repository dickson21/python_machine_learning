"""
Subject : data set binarization
Status : 
 cmd console : NotOK

Detail :
Binarization
As the name suggests, this is the technique with the help of which we can make our data binary. 
We can use a binary threshold for making our data binary. 
The values above that threshold value will be converted to 1 and below that threshold will be converted to 0.
"""

from pandas import read_csv
from sklearn.preprocessing import Binarizer
#path = r'C:\pima-indians-diabetes.csv'
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values

binarizer = Binarizer(threshold=0.5).fit(array)
Data_binarized = binarizer.transform(array)

print ("\nBinary data:\n", Data_binarized [0:5])

"""
Binary data:
 [[1. 1. 1. 1. 0. 1. 1. 1. 1.]
 [1. 1. 1. 1. 0. 1. 0. 1. 0.]
 [1. 1. 1. 0. 0. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1. 1. 0. 1. 0.]
 [0. 1. 1. 1. 1. 1. 1. 1. 1.]]
"""