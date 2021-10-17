"""
Subject : Performance Improvement with Algorithm Tuning
Created : 2020-08-29
Author : dickson.cheng
Status : OK

Detail :
Performance Improvement with Algorithm Tuning
As we know that ML models are parameterized in such a way that their behavior can be adjusted for a specific problem. 
Algorithm tuning means finding the best combination of these parameters so that the performance of ML model can be improved. 
This process sometimes called hyperparameter optimization and 
the parameters of algorithm itself are called hyperparameters and 
coefficients found by ML algorithm are called parameters

Ref :
Improving Performance of ML Model(Contd..)
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_improving_performance_of_ml_model.htm


"""
import numpy
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

#path = r"C:\pima-indians-diabetes.csv"
path=r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names = headernames)
array = data.values
X = array[:,0:8]
Y = array[:,8]

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha = alphas)

model = Ridge()
grid = GridSearchCV(estimator = model, param_grid = param_grid)
grid.fit(X, Y)

print(grid.best_score_)
print(grid.best_estimator_.alpha)

"""
0.2796175593129722
1.0
"""

