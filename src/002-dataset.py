"""
Machine Learning in tutorialpoint (*)
https://www.tutorialspoint.com/machine_learning_with_python/index.htm
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_quick_guide.htm
"""

from pandas import read_csv
path = r"C:\myprogram\python\machine_learning\data\iris.csv" #OK
data = read_csv(path)
print(data.shape)
print(data.dtypes)
print(data[:3])

"""
(150, 5)
   sepal.length  sepal.width  petal.length  petal.width variety
0           5.1          3.5           1.4          0.2  Setosa
1           4.9          3.0           1.4          0.2  Setosa
2           4.7          3.2           1.3          0.2  Setosa
"""