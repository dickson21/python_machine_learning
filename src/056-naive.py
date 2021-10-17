"""
Subject : Naive bayes agm
Created : 2020-08-28
Author : dickson.cheng
Status : OK
Detail :
Introduction to Naïve Bayes Algorithm
Naïve Bayes algorithms is a classification technique based on applying Bayes’ theorem with a strong assumption that 
all the predictors are independent to each other. 
In simple words, 
the assumption is that the presence of a feature in a class is independent to the presence of any other feature in the same class. 
For example, a phone may be considered as smart if it is having touch screen, internet facility, good camera etc. 
Though all these features are dependent on each other, 
they contribute independently to the probability of that the phone is a smart phone.

Applications of Naïve Bayes classification
------------------------------------------
The following are some common applications of Naïve Bayes classification −
01# Real-time prediction − 
Due to its ease of implementation and fast computation, it can be used to do prediction in real-time.

02# Multi-class prediction − 
Naïve Bayes classification algorithm can be used to predict posterior probability of multiple classes of target variable.

03# Text classification − 
Due to the feature of multi-class prediction, Naïve Bayes classification algorithms are well suited for text classification. 
That is why it is also used to solve problems like spam-filtering and sentiment analysis.

04# Recommendation system − 
Along with the algorithms like collaborative filtering, 
Naïve Bayes makes a Recommendation system which can be used to filter unseen information and 
to predict weather a user would like the given resource or not.

Ref :
Naive agm
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_classification_algorithms_naive_bayes.htm
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB

X, y = make_blobs(300, 2, centers = 2, random_state = 2, cluster_std = 1.5)
plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'summer')


model_GNB = GaussianNB()
model_GNB.fit(X, y)

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model_GNB.predict(Xnew)

plt.scatter(X[:, 0], X[:, 1], c = y, s = 50, cmap = 'summer')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c = ynew, s = 20, cmap = 'summer', alpha = 0.1)
plt.axis(lim)

yprob = model_GNB.predict_proba(Xnew)
yprob[-10:].round(3)

"""
array([[0.998, 0.002],
   [1. , 0. ],
   [0.987, 0.013],
   [1. , 0. ],
   [1. , 0. ],
   [1. , 0. ],
   [1. , 0. ],
   [1. , 0. ],
   [0. , 1. ],
   [0.986, 0.014]])

"""