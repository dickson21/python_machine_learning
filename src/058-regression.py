"""
Subject : Regression
Created : 2020-08-28
Author : dickson.cheng
Status : OK

Detail :
Introduction to Regression
Regression is another important and broadly used statistical and machine learning tool. 
The key objective of regression-based tasks is to predict output labels or responses which are continues numeric values, 
for the given input data. 
The output will be based on what the model has learned in training phase. 
Basically, regression models use the input data features (independent variables) and 
their corresponding continuous numeric output values (dependent or outcome variables) to learn specific association 
between inputs and corresponding outputs.

Types of ML Regression Algorithms
The most useful and popular ML regression algorithm is Linear regression algorithm which further divided into two types namely −
01# Simple Linear Regression algorithm
02# Multiple Linear Regression algorithm.
We will discuss about it and implement it in Python in the next chapter.

Applications
The applications of ML regression algorithms are as follows −
01# Forecasting or Predictive analysis − One of the important uses of regression is forecasting or predictive analysis. For example, we can forecast GDP, oil prices or in simple words the quantitative data that changes with the passage of time.

02# Optimization − 
We can optimize business processes with the help of regression. 
For example, a store manager can create a statistical model to understand the peek time of coming of customers.

03# Error correction − 
In business, taking correct decision is equally important as optimizing the business process. 
Regression can help us to take correct decision as well in correcting the already implemented decision.

04# Economics − 
It is the most used tool in economics. 
We can use regression to predict supply, demand, consumption, inventory investment etc.

05# Finance − 
A financial company is always interested in minimizing the risk portfolio and 
want to know the factors that affects the customers. 
All these can be predicted with the help of regression model.

Ref:
linear regression (tutorialspoint)
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_regression_algorithms_overview.htm

Introduction to Linear Regression
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_regression_algorithms_linear_regression.htm

github diabetes dataset
https://raw.githubusercontent.com/psmathur/simple-linear-regression-using-python-only/master/diabetes.csv

Linear Regression Datasets
https://people.sc.fsu.edu/~jburkardt/datasets/regression/regression.html

"""

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

#input = r'C:\linear.txt'
input = r'C:\myprogram\python\machine_learning\data\diabetes.csv'
num_training = 100

input_data = np.loadtxt(input, delimiter=',')
X, y = input_data[:, :-1], input_data[:, -1]

training_samples = int(0.6 * len(X))
testing_samples = len(X) - num_training
X_train, y_train = X[:training_samples], y[:training_samples]
X_test, y_test = X[training_samples:], y[training_samples:]

reg_linear = linear_model.LinearRegression()
reg_linear.fit(X_train, y_train)
y_test_pred = reg_linear.predict(X_test)

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, y_test_pred, color = 'black', linewidth = 2)
plt.xticks(())
plt.yticks(())
plt.show()

print("Regressor model performance:")
print("Mean absolute error(MAE) =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error(MSE) =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

"""
linear.txt
Regressor model performance:
Mean absolute error(MAE) = 1.78
Mean squared error(MSE) = 3.89
Median absolute error = 2.01
Explain variance score = -0.09
R2 score = -0.09

diabetes.csv
Regressor model performance:
Mean absolute error(MAE) = 99.53
Mean squared error(MSE) = 10659.57
Median absolute error = 94.91
Explain variance score = -0.69
R2 score = -22.95

"""
