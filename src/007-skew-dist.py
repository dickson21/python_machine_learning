
"""
Reviewing Skew of Attribute Distribution
Skewness may be defined as the distribution that is assumed to be Gaussian 
but appears distorted or shifted in one direction or another, or either to the left or right. 
Reviewing the skewness of attributes is one of the important tasks due to following reasons −

Presence of skewness in data requires the correction at data preparation stage so that we can get more accuracy from our model.

Most of the ML algorithms assumes that data has a Gaussian distribution i.e. either normal of bell curved data.

In Python, we can easily calculate the skew of each attribute by using skew() function on Pandas DataFrame.
"""

from pandas import read_csv
#path = r"C:\pima-indians-diabetes.csv"
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
print(data.skew())

"""
From the above output, positive or negative skew can be observed. 
If the value is closer to zero, then it shows less skew.

preg     0.901674
plas     0.173754
pres    -1.843608
skin     0.109372
test     2.272251
mass    -0.428982
pedi     1.919911
age      1.129597
class    0.635017
dtype: float64
"""