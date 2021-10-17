"""
Subject : automatic workflow
Created : 2020-08-28
Author : dickson.cheng
Status : OK


Ref :
automatic workflow
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_pipelines_automatic_workflows.htm
"""

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#path = r"C:\pima-indians-diabetes.csv"
path=r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames)
array = data.values

X = array[:,0:8]
Y = array[:,8]

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('lda', LinearDiscriminantAnalysis()))
model = Pipeline(estimators)

kfold = KFold(n_splits = 20, random_state = 7)
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())

"""
0.7790148448043184
"""

