"""
Subject : Organizing data into training & testing sets
Created : 2020-08-27
Author : dickson.cheng
Status : OK

Detail :
Step 3: Organizing data into training & testing sets
As we need to test our model on unseen data, we will divide our dataset into two parts: a training set and a test set. 
We can use train_test_split() function of sklearn python package to split the data into sets. 
The following command will import the function 

"""
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(features,labels,test_size = 0.40, random_state = 42)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

model = gnb.fit(train, train_labels)
preds = gnb.predict(test)
print(preds)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels,preds))

"""
[1 0 0 1 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 1 1 1 1 1 1 0
 1 0 1 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 0 0 1 1 0 0 1 1 1 0 0 1 1 0 0 1 0
 1 1 1 1 1 1 0 1 1 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 1 0
 1 1 0 0 0 1 1 1 0 0 1 1 0 1 0 0 1 1 0 0 0 1 1 1 0 1 1 0 0 1 0 1 1 0 1 0 0
 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 1 0 1 1 1 1 1 1 0 0
 0 1 1 0 1 0 1 1 1 1 0 1 1 0 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1
 0 0 1 1 0 1]
0.9517543859649122
"""