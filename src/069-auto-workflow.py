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
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest

#path = r"C:\pima-indians-diabetes.csv"
path=r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames)
array = data.values

X = array[:,0:8]
Y = array[:,8]

features = []
features.append(('pca', PCA(n_components=3)))
features.append(('select_best', SelectKBest(k=6)))
feature_union = FeatureUnion(features)

estimators = []
estimators.append(('feature_union', feature_union))
estimators.append(('logistic', LogisticRegression()))
model = Pipeline(estimators)

kfold = KFold(n_splits = 20, random_state = 7)
results = cross_val_score(model, X, Y, cv = kfold)
print(results.mean())

"""
0.7789811066126855
"""