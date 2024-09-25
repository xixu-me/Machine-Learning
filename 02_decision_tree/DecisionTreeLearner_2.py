import sys
sys.path.insert(1, '../')
from utils.utils import *


class DecisionFork:
    """
    A fork of a decision tree holds an attribute to test, and a dict
    of branches, one for each of the attribute's values.
    """

    def __init__(self, attr, attr_name=None, default_child=None, branches=None):
        """Initialize by saying what attribute this node tests."""
        self.attr = attr
        self.attr_name = attr_name or attr
        self.default_child = default_child
        self.branches = branches or {}

    def __call__(self, example):
        """Given an example, classify it using the attribute and the branches."""
        attr_val = example[self.attr]
        if attr_val in self.branches:
            return self.branches[attr_val](example)
        else:
            # return default class when attribute is unknown
            return self.default_child(example)

    def add(self, val, subtree):
        """Add a branch. If self.attr = val, go to the given subtree."""
        self.branches[val] = subtree

    def display(self, indent=0):
        name = self.attr_name
        print('Test', name)
        for (val, subtree) in self.branches.items():
            print(' ' * 4 * indent, name, '=', val, '==>', end=' ')
            subtree.display(indent + 1)

    def __repr__(self):
        return 'DecisionFork({0!r}, {1!r}, {2!r})'.format(self.attr, self.attr_name, self.branches)


class DecisionLeaf:
    """A leaf of a decision tree holds just a result."""

    def __init__(self, result):
        self.result = result

    def __call__(self, example):
        return self.result

    def display(self):
        print('RESULT =', self.result)

    def __repr__(self):
        return repr(self.result)


class DecisionTreeLearner:
    """DecisionTreeLearner: based on information gain"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.tree = self.decision_tree_learning(dataset.examples, dataset.inputs)

    def decision_tree_learning(self, examples, attrs, parent_examples=()):
        if len(examples) == 0:
            return self.plurality_value(parent_examples)
        if self.all_same_class(examples):
            return DecisionLeaf(examples[0][self.dataset.target])
        if len(attrs) == 0:
            return self.plurality_value(examples)
        A = self.choose_attribute(attrs, examples)
        tree = DecisionFork(A, self.dataset.attr_names[A], self.plurality_value(examples))
        for (v_k, exs) in self.split_by(A, examples):
            subtree = self.decision_tree_learning(exs, remove_all(A, attrs), examples)
            tree.add(v_k, subtree)
        return tree

    def plurality_value(self, examples):
        """
        Return the most popular target value for this set of examples.
        (If target is binary, this is the majority; otherwise plurality).
        """
        popular = argmax_random_tie(self.dataset.values[self.dataset.target],
                                    key=lambda v: self.count(self.dataset.target, v, examples))
        return DecisionLeaf(popular)

    def count(self, attr, val, examples):
        """Count the number of examples that have example[attr] = val."""
        return sum(e[attr] == val for e in examples)

    def all_same_class(self, examples):
        """Are all these examples in the same target class?"""
        class0 = examples[0][self.dataset.target]
        return all(e[self.dataset.target] == class0 for e in examples)

    def choose_attribute(self, attrs, examples):
        """Choose the attribute with the highest information gain."""
        return argmax_random_tie(attrs, key=lambda a: self.information_gain(a, examples))

    def information_gain(self, attr, examples):
        """Return the expected reduction in entropy from splitting by attr."""
        raise NotImplementedError

    def split_by(self, attr, examples):
        """Return a list of (val, examples) pairs for each val of attr."""
        return [(v, [e for e in examples if e[attr] == v]) for v in self.dataset.values[attr]]

    def predict(self, x):
        return self.tree(x)

    def __call__(self, x):
        return self.predict(x)


def information_content(values):
    """Number of bits to represent the probability distribution in values."""
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
