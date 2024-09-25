import sys
sys.path.insert(1, '../')
from utils.utils import *
from utils.dataset4learners import *


class LinearClassifier:
    def learn(self, learning_rate, epochs):
        raise NotImplementedError
        
    def predict(self, x):
        raise NotImplementedError
    
    
class PerceptionLinearLearner(LinearClassifier):
    """
    Perception linear classifier: hard threshold
    """
    def __init__(self, dataset, learning_rate=0.01, epochs=100):
        self.idx_i = dataset.inputs
        self.idx_t = dataset.target
        self.examples = dataset.examples
        self.num_examples = len(self.examples)
        # initialize random weights
        self.w = random_weights(min_value=-0.5, max_value=0.5, num_weights=len(self.idx_i) + 1)
        # learning loop
        self.learn(learning_rate, epochs)
        
    def learn(self, learning_rate, epochs):
        """ learning loop """
        def loss(example, w, idx_i, idx_t):
            """ error: difference between estimation and true value """
            raise NotImplementedError
        
        def update(w, learning_rate, err, X_col, num_examples):
            """ update weights """
            raise NotImplementedError

        def homogeneous(num_examples):
            """ build homogeneous coordinates """
            raise NotImplementedError

        raise NotImplementedError

    def predict(self, x):
        """ make prediction """
        return int(np.dot(self.w, [1] + x))


class LogisticLinearLeaner(LinearClassifier):
    def __init__(self, dataset, learning_rate=0.01, epochs=100):
        self.idx_i = dataset.inputs
        self.idx_t = dataset.target
        self.examples = dataset.examples
        self.num_examples = len(self.examples)
        # initialize random weights
        self.w = random_weights(min_value=-0.5, max_value=0.5, num_weights=len(self.idx_i) + 1)
        # learning loop
        self.learn(learning_rate, epochs)
        
    def learn(self, learning_rate, epochs):
        """ learning loop """
        def loss(example, w, idx_i, idx_t, h):
            """ error: difference between estimation and true value """
            raise NotImplementedError
        
        def update(w, learning_rate, err, h, X_col, num_examples):
            """ update weights """
            raise NotImplementedError

        def homogeneous(num_examples):
            """ build homogeneous coordinates """
            raise NotImplementedError

        raise NotImplementedError

    def predict(self, x):
        """ make prediction """
        return int(np.dot(self.w, [1] + x))


if __name__ == "__main__":
    iris = DataSet(name="iris")
    iris.classes_to_numbers()
    
    tests = [([5, 3, 1, 0.1], 0),
             ([5, 3.5, 1, 0], 0),
             ([6, 3, 4, 1.1], 1),
             ([6, 2, 3.5, 1], 1),
             ([7.5, 4, 6, 2], 2),
             ([7, 3, 6, 2.5], 2)]
    
    print(f'===================\nperceptron:')
    perceptron = PerceptionLinearLearner(iris)
    g = grade_learner(perceptron, tests)
    print(f'  learner grade: {g}')
    # assert g > 1. / 2
    e = err_ratio(perceptron, iris)
    print(f'  error ration: {e}')
    # assert e < 0.4
    
    print(f'===================\nlogistic:')
    logisticer = LogisticLinearLeaner(iris)
    g = grade_learner(logisticer, tests)
    print(f'  learner grade: {g}')
    # assert g > 1. / 2
    e = err_ratio(logisticer, iris)
    print(f'  error ration: {e}')
    # assert e < 0.4
