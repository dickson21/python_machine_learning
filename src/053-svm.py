"""
Subject : Support vector machines (SVM)
Created : 2020-08-27
Author : dickson.cheng
Status : OK

Detail :
Introduction to SVM
Support vector machines (SVMs) are powerful yet flexible supervised machine learning algorithms which are used both for classification and regression. But generally, they are used in classification problems. In 1960s, SVMs were first introduced but later they got refined in 1990. 
SVMs have their unique way of implementation as compared to other machine learning algorithms. 
Lately, they are extremely popular because of their ability to handle multiple continuous and categorical variables.

Working of SVM
An SVM model is basically a representation of different classes in a hyperplane in multidimensional space. 
The hyperplane will be generated in an iterative manner by SVM so that the error can be minimized. 
The goal of SVM is to divide the datasets into classes to find a maximum marginal hyperplane (MMH)
"""

"""

ref :
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_classification_algorithms_support_vector_machine.htm
"""

import pandas as pd
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]

C=1.0

Svc_classifier = svm.SVC(kernel = 'rbf', gamma ='auto',C = C).fit(X, y)
Z = svc_classifier.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap = plt.cm.tab10, alpha = 0.3)
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('Support Vector Classifier with rbf kernel')

"""
Text(0.5, 1.0, 'Support Vector Classifier with rbf kernel')
"""

