"""
Subject : Feature Importance
Created : 2020-08-27
Author : dickson.cheng
Status : OK

Detail :
Feature Importance
As the name suggests, 
feature importance technique is used to choose the importance features. 
It basically uses a trained supervised classifier to select features. 
We can implement this feature selection technique with the help of ExtraTreeClassifier class of scikit-learn Python library
"""

from pandas import read_csv
from sklearn.ensemble import ExtraTreesClassifier
#path = r'C:\Desktop\pima-indians-diabetes.csv'
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

model = ExtraTreesClassifier()
model.fit(X, Y)
print(model.feature_importances_)

"""
[ 0.11070069 0.2213717 0.08824115 0.08068703 0.07281761 0.14548537 0.12654214 0.15415431]
"""