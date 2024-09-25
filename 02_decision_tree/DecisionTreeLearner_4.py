import sys
sys.path.insert(1, '../')
from utils.utils import *


class DecisionFork:
    """
    A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values.
    """
    raise NotImplementedError


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""
    raise NotImplementedError


class DecisionTreeLearner:
    """DecisionTreeLearner: based on information gain"""
    raise NotImplementedError


if __name__ == "__main__":
    from utils.dataset4learners import *

    iris = DataSet(name="iris")
    DTL = DecisionTreeLearner(iris)
    print(f'DTL.predict([5, 3, 1, 0.1]): {DTL.predict([5, 3, 1, 0.1])}')
    assert DTL.predict([5, 3, 1, 0.1]) == 'setosa'
    print(f'DTL.predict([6, 5, 3, 1.5]): {DTL.predict([6, 5, 3, 1.5])}')
    assert DTL.predict([6, 5, 3, 1.5]) == 'versicolor'
    print(f'DTL.predict([7.5, 4, 6, 2]): {DTL.predict([7.5, 4, 6, 2])}')
    assert DTL.predict([7.5, 4, 6, 2]) == 'virginica'
