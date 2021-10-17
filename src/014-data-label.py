

"""

Detail :
Data Labeling
We discussed the importance of good fata for ML algorithms as well as 
some techniques to pre-process the data before sending it to ML algorithms. One more aspect in this regard is data labeling. 
t is also very important to send the data to ML algorithms having proper labeling. 
For example, in case of classification problems, lot of labels in the form of words, numbers etc. are there on the data.

ref :
prepare data
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_preparing_data.htm
"""

import numpy as np
from sklearn import preprocessing

input_labels = ['red','black','red','green','black','yellow','white']
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)
test_labels = ['green','red','black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))
encoded_values = [3,0,4,1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("\nDecoded labels =", list(decoded_list))

"""
Labels = ['green', 'red', 'black']
Encoded values = [1, 2, 0]

Encoded values = [3, 0, 4, 1]

Decoded labels = ['white', 'black', 'yellow', 'green']
"""

