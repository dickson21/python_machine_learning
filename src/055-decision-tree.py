"""
Subject : decision tree visualise
Created : 2020-08-27
Author : dickson.cheng
Status : OK

ref:
https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_classification_algorithms_decision_tree.htm
"""

from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
   special_characters=True,feature_names = feature_cols,class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Pima_diabetes_Tree.png')
Image(graph.create_png())


"""
(base) C:\Users\dickson.cheng>conda install pydotplus
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: C:\Users\dickson.cheng\Anaconda3

  added / updated specs:
    - pydotplus


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    conda-4.8.4                |           py37_0         2.9 MB
    graphviz-2.38              |       hfd603c8_2        29.3 MB
    pydotplus-2.0.2            |             py_3          23 KB
    ------------------------------------------------------------
                                           Total:        32.2 MB

The following NEW packages will be INSTALLED:

  graphviz           pkgs/main/win-64::graphviz-2.38-hfd603c8_2
  pydotplus          pkgs/main/noarch::pydotplus-2.0.2-py_3

The following packages will be UPDATED:

  conda                                        4.8.3-py37_0 --> 4.8.4-py37_0


Proceed ([y]/n)? y


Downloading and Extracting Packages
conda-4.8.4          | 2.9 MB    | ############################################################################ | 100%
pydotplus-2.0.2      | 23 KB     | ############################################################################ | 100%
graphviz-2.38        | 29.3 MB   | ############################################################################ | 100%
Preparing transaction: done
Verifying transaction: done
Executing transaction: done

"""