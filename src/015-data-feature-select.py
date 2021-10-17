
"""
Subject : data feature selection
Created : 2020-08-27
Author : dickson.cheng
Status :  OK (JupyterLab)
Detail :
Importance of Data Feature Selection
The performance of machine learning model is directly proportional to the data features used to train it. 
The performance of ML model will be affected negatively if the data features provided to it are irrelevant. 
On the other hand, 
use of relevant data features can increase the accuracy of your ML model especially linear and logistic regression.

Now the question arise that what is automatic feature selection? 
It may be defined as the process with the help of which we select those features in our data 
that are most relevant to the output or prediction variable in which we are interested. 
It is also called attribute selection.

The following are some of the benefits of automatic feature selection before modeling the data −
01# Performing feature selection before data modeling will reduce the overfitting.
02# Performing feature selection before data modeling will increases the accuracy of ML model.
03# Performing feature selection before data modeling will reduce the training time

Feature Selection Techniques
The followings are automatic feature selection techniques that we can use to model ML data in Python −

Univariate Selection
This feature selection technique is very useful in selecting those features, 
with the help of statistical testing, having strongest relationship with the prediction variables. 
We can implement univariate feature selection technique with the help of SelectKBest0class of scikit-learn Python library.

"""

from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#path = r'C:\pima-indians-diabetes.csv'
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(path, names=names)
array = dataframe.values

X = array[:,0:8]
Y = array[:,8]

test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X,Y)

set_printoptions(precision=2)
print(fit.scores_)
featured_data = fit.transform(X)
print ("\nFeatured data:\n", featured_data[0:4])


"""
[ 111.52 1411.89   17.61   53.11 2175.57  127.67    5.39  181.3 ]

Featured data:
 [[148.    0.   33.6  50. ]
 [ 85.    0.   26.6  31. ]
 [183.    0.   23.3  32. ]
 [ 89.   94.   28.1  21. ]]
"""
