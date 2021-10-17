
"""
Reviewing Class Distribution
Class distribution statistics is useful in classification problems where we need to know the balance of class values. 
It is important to know class value distribution because if we have highly imbalanced class distribution 
i.e. 
one class is having lots more observations than other class, 
then it may need special handling at data preparation stage of our ML project. 
We can easily get class distribution in Python with the help of Pandas DataFrame

"""

from pandas import read_csv
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
count_class = data.groupby('class').size()
print(count_class)

"""
class
0    500
1    268
dtype: int64
""""