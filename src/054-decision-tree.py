"""
Subject : Decision tree
Created : 2020-08-27
Author : dickson.cheng
Status : OK

Detail :
Introduction to Decision Tree
In general, Decision tree analysis is a predictive modelling tool that can be applied across many areas. 
Decision trees can be constructed by an algorithmic approach that can split the dataset in different ways based on different conditions. 
Decisions trees are the most powerful algorithms that falls under the category of supervised algorithms.

They can be used for both classification and regression tasks. 
The two main entities of a tree are decision nodes, 
where the data is split and leaves, where we got outcome. 
The example of a binary tree for predicting whether a person is fit or unfit providing various information 
like age, eating habits and exercise habits, is given below

In the above decision tree, the question are decision nodes and final outcomes are leaves. We have the following two types of decision trees.
01# Classification decision trees − 
In this kind of decision trees, the decision variable is categorical. 
The above decision tree is an example of classification decision tree.

02# Regression decision trees − 
In this kind of decision trees, the decision variable is continuous.

ref:
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_classification_algorithms_decision_tree.htm
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv", header = None, names = col_names)
print(pima.head())

feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
X = pima[feature_cols] # Features
y = pima.label # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)



"""
	pregnant  glucose  bp  skin  insulin   bmi  pedigree  age  label
-------------------------------------------------------------------------
0         6      148  	72    35        0  33.6     0.627   50      1
1         1       85  	66    29        0  26.6     0.351   31      0
2         8      183  	64     0        0  23.3     0.672   32      1
3         1       89  	66    23       94  28.1     0.167   21      0
4         0      137  	40    35      168  43.1     2.288   33      1
-------------------------------------------------------------------------

Confusion Matrix:
[[113  33]
 [ 43  42]]
Classification Report:
              precision    recall  f1-score   support

           0       0.72      0.77      0.75       146
           1       0.56      0.49      0.53        85

    accuracy                           0.67       231
   macro avg       0.64      0.63      0.64       231
weighted avg       0.66      0.67      0.67       231

Accuracy: 0.670995670995671
"""