"""
Subject : Building a Classifier in Python
Created : 2020-08-27
Author : dickson.cheng
Status : OK

Detail :
Building a Classifier in Python
Scikit-learn, a Python library for machine learning can be used to build a classifier in Python. 
The steps for building a classifier in Python are as follows
"""

import sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
print(label_names)
print(feature_names[0])
print(feature_names[1])
print(features[0])
print(features[1])

"""
['malignant' 'benign']
mean radius
mean texture
[1.799e+01 1.038e+01 1.228e+02 1.001e+03 1.184e-01 2.776e-01 3.001e-01
 1.471e-01 2.419e-01 7.871e-02 1.095e+00 9.053e-01 8.589e+00 1.534e+02
 6.399e-03 4.904e-02 5.373e-02 1.587e-02 3.003e-02 6.193e-03 2.538e+01
 1.733e+01 1.846e+02 2.019e+03 1.622e-01 6.656e-01 7.119e-01 2.654e-01
 4.601e-01 1.189e-01]

[2.057e+01 1.777e+01 1.329e+02 1.326e+03 8.474e-02 7.864e-02 8.690e-02
7.017e-02  1.812e-01 5.667e-02 5.435e-01 7.339e-01 3.398e+00 7.408e+01
5.225e-03  1.308e-02 1.860e-02 1.340e-02 1.389e-02 3.532e-03 2.499e+01
2.341e+01  1.588e+02 1.956e+03 1.238e-01 1.866e-01 2.416e-01 1.860e-01
2.750e-01  8.902e-02]
"""