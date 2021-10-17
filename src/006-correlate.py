
"""
Reviewing Correlation between Attributes
The relationship between two variables is called correlation. 
In statistics, the most common method for calculating correlation is Pearson’s Correlation Coefficient. 
It can have three values as follows −

Coefficient value = 1 − It represents full positive correlation between variables.

Coefficient value = -1 − It represents full negative correlation between variables.

Coefficient value = 0 − It represents no correlation at all between variables.

It is always good for us to review the pairwise correlations of the attributes in our dataset before using it into ML project 
because some machine learning algorithms such as linear regression and logistic regression will perform poorly 
if we have highly correlated attributes. 
In Python, we can easily calculate a correlation matrix of dataset attributes with the help of corr() function on Pandas DataFrame.
"""

"""
Ref:
Machine Learning in tutorialpoint (*)
https://www.tutorialspoint.com/machine_learning_with_python/index.htm

Quick guide
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_quick_guide.htm
"""


from pandas import read_csv
from pandas import set_option
#path = r"C:\pima-indians-diabetes.csv"
path = r"C:\myprogram\python\machine_learning\data\pima-indians-diabetes.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=names)
set_option('display.width', 100)
set_option('precision', 2)
correlations = data.corr(method='pearson')
print(correlations)

"""
       preg  plas  pres  skin  test  mass  pedi   age  class
preg   1.00  0.13  0.14 -0.08 -0.07  0.02 -0.03  0.54   0.22
plas   0.13  1.00  0.15  0.06  0.33  0.22  0.14  0.26   0.47
pres   0.14  0.15  1.00  0.21  0.09  0.28  0.04  0.24   0.07
skin  -0.08  0.06  0.21  1.00  0.44  0.39  0.18 -0.11   0.07
test  -0.07  0.33  0.09  0.44  1.00  0.20  0.19 -0.04   0.13
mass   0.02  0.22  0.28  0.39  0.20  1.00  0.14  0.04   0.29
pedi  -0.03  0.14  0.04  0.18  0.19  0.14  1.00  0.03   0.17
age    0.54  0.26  0.24 -0.11 -0.04  0.04  0.03  1.00   0.24
class  0.22  0.47  0.07  0.07  0.13  0.29  0.17  0.24   1.00
"""