"""
Subject : introduction to performance metrics
Created : 2020-08-28
Author : dickson.cheng
Status : OK


Ref :
performance metrics
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_algorithms_performance_metrics.htm
"""

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
X_actual = [5, -1, 2, 10]
Y_predic = [3.5, -0.9, 2, 9.9]
print ('R Squared =',r2_score(X_actual, Y_predic))
print ('MAE =',mean_absolute_error(X_actual, Y_predic))
print ('MSE =',mean_squared_error(X_actual, Y_predic))

"""
R Squared = 0.9656060606060606
MAE = 0.42499999999999993
MSE = 0.5674999999999999
"""