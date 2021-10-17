"""
Subject : Performance Improvement with Ensembles
Created : 2020-08-29
Author : dickson.cheng
Status : OK


Detail :
Performance Improvement with Ensembles
Ensembles can give us boost in the machine learning result by combining several models. 
Basically, ensemble models consist of several individually trained supervised learning models and 
their results are merged in various ways to achieve better predictive performance compared to a single model. 
Ensemble methods can be divided into following two groups

Ref : 
Improving Performance of ML Models
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_improving_performance_of_ml_models.htm
"""

from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

#path = r"C:\pima-indians-diabetes.csv"
path=r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]

kfold = KFold(n_splits = 10, random_state = 7)

estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv = kfold)
print(results.mean())

"""
0.7382262474367738
"""