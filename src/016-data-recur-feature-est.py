"""
Subject : Recursive Feature Elimination
Created : 2020-08-27
Author : dickson.cheng
Status : OK

Detail :
Recursive Feature Elimination
As the name suggests, 
RFE (Recursive feature elimination) feature selection technique removes the attributes recursively and 
builds the model with remaining attributes. 
We can implement RFE feature selection technique with the help of RFE class of scikit-learn Python library


Ref:
feature selection in python
https://www.datacamp.com/community/tutorials/feature-selection-python

recursive feature selection
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_data_feature_selection.htm


"""

from pandas import read_csv
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
#path = r'C:\pima-indians-diabetes.csv'
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Number of Features: %d" %(fit.n_features_)) 
print("Selected Features: %s" %(fit.support_))
print("Feature Ranking: %s" %(fit.ranking_))

"""
Number of Features: 3
Selected Features: [ True False False False False True True False]
Feature Ranking: [1 2 3 5 6 1 1 4]

"""

