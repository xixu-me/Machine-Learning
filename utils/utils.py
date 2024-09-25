### system related operations
import os
from inspect import getsource
from IPython.display import HTML
from IPython.display import display


def psource(*functions):
    """Print the source code for the given function(s)."""
    source_code = '\n\n'.join(getsource(fn) for fn in functions)
    try:
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import PythonLexer
        from pygments import highlight

        display(HTML(highlight(source_code, PythonLexer(), HtmlFormatter(full=True))))

    except ImportError:
        print(source_code)

        
def parse_csv(input, delim=','):
    r"""
    Input is a string consisting of lines, each line has comma-delimited
    fields. Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.
    >>> parse_csv('1, 2, 3 \n 0, 2, na')
    [[1, 2, 3], [0, 2, 'na']]
    """
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]


def open_data(name, mode='r'):
    data_root = os.path.dirname(__file__)
    data_file = os.path.join(data_root, *[os.pardir, 'data', name])

    return open(data_file, mode=mode)


### math and data structure
import numpy as np
import math
from statistics import mean, stdev
import random


def normalize(dist):
    """Multiply each number by a constant such that the sum is 1.0"""
    if isinstance(dist, dict):
        total = sum(dist.values())
        for key in dist:
            dist[key] = dist[key] / total
            assert 0 <= dist[key] <= 1  # probabilities must be between 0 and 1
        return dist
    total = sum(dist)
    return [(n / total) for n in dist]


def random_weights(min_value, max_value, num_weights):
    return [random.uniform(min_value, max_value) for _ in range(num_weights)]


def sigmoid(x):
    """Return activation value of x with sigmoid function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(value):
    return value * (1 - value)


def remove_all(item, seq):
    """Return a copy of seq (or string) with all occurrences of item removed."""
    if isinstance(seq, str):
        return seq.replace(item, '')
    elif isinstance(seq, set):
        rest = seq.copy()
        rest.remove(item)
        return rest
    else:
        return [x for x in seq if x != item]
    
    
def unique(seq):
    """Remove duplicate elements from seq. Assumes hashable elements."""
    return list(set(seq))


def num_or_str(x):  # TODO: rename as `atom`
    """The argument is a string; convert to a number if
       possible, or strip it."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()
        

def euclidean_distance(x, y):
    return np.sqrt(sum((_x - _y) ** 2 for _x, _y in zip(x, y)))


def manhattan_distance(x, y):
    return sum(abs(_x - _y) for _x, _y in zip(x, y))


def hamming_distance(x, y):
    return sum(_x != _y for _x, _y in zip(x, y))


def rms_error(x, y):
    return np.sqrt(ms_error(x, y))


def ms_error(x, y):
    return mean((x - y) ** 2 for x, y in zip(x, y))


def mean_error(x, y):
    return mean(abs(x - y) for x, y in zip(x, y))


def mean_boolean_error(x, y):
    return mean(_x != _y for _x, _y in zip(x, y))


identity = lambda x: x


def argmin_random_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return min(shuffled(seq), key=key)


def argmax_random_tie(seq, key=identity):
    """Return an element with highest fn(seq[i]) score; break ties at random."""
    return max(shuffled(seq), key=key)


def shuffled(iterable):
    """Randomly shuffle a copy of iterable."""
    items = list(iterable)
    random.shuffle(items)
    return items


### learner related operations
class Activation:

    def function(self, x):
        return NotImplementedError

    def derivative(self, x):
        return NotImplementedError

    def __call__(self, x):
        return self.function(x)


class Sigmoid(Activation):

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, value):
        return value * (1 - value)

    
def err_ratio(learner, dataset, examples=None):
    """
    Return the proportion of the examples that are NOT correctly predicted.
    verbose - 0: No output; 1: Output wrong; 2 (or greater): Output correct
    """
    examples = examples or dataset.examples
    if len(examples) == 0:
        return 0.0
    right = 0
    for example in examples:
        desired = example[dataset.target]
        output = learner.predict(dataset.sanitize(example))
        if output == desired:
            right += 1
    return 1 - (right / len(examples))


def grade_learner(learner, tests):
    """
    Grades the given learner based on how many tests it passes.
    tests is a list with each element in the form: (values, output).
    """
    # for X, y in tests:
    #     print(learner.predict(X), y)
    return mean([int(learner.predict(X) == y) for X, y in tests])
    