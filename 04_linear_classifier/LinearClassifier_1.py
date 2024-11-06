import sys

sys.path.insert(1, "../")
from utils.dataset4learners import *
from utils.utils import *


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
        self.w = random_weights(
            min_value=-0.5, max_value=0.5, num_weights=len(self.idx_i) + 1
        )
        # learning loop
        self.learn(learning_rate, epochs)

    def learn(self, learning_rate, epochs):
        """learning loop"""

        def loss(example, w, idx_i, idx_t):
            """error: difference between estimation and true value"""
            # Get input features and true target value
            x = [example[i] for i in idx_i]
            y = example[idx_t]

            # Add bias term (1) to input features
            x = [1] + x

            # Calculate predicted value using dot product
            prediction = np.dot(w, x)

            # Return difference between prediction and true value
            return prediction - y

        def update(w, learning_rate, err, X_col, num_examples):
            """update weights"""
            for i in range(len(w)):
                w[i] = w[i] - learning_rate * (np.dot(err, X_col[i]) / num_examples)

        def homogeneous(num_examples):
            """build homogeneous coordinates"""
            # Initialize matrix with zeros
            X = np.zeros((len(self.w), num_examples))

            # Fill the matrix with features
            for i, example in enumerate(self.examples):
                # First row is bias terms (all 1s)
                X[0][i] = 1
                # Remaining rows are feature values
                for j, idx in enumerate(self.idx_i, 1):
                    X[j][i] = example[idx]

            return X

        X_col = homogeneous(self.num_examples)
        for epoch in range(epochs):
            err = []
            # pass over all examples
            for example in self.examples:
                err.append(loss(example, self.w, self.idx_i, self.idx_t))

            # update weights
            update(self.w, learning_rate, err, X_col, self.num_examples)

    def predict(self, x):
        """make prediction"""
        return int(np.dot(self.w, [1] + x))


class LogisticLinearLeaner(LinearClassifier):
    def __init__(self, dataset, learning_rate=0.01, epochs=100):
        self.idx_i = dataset.inputs
        self.idx_t = dataset.target
        self.examples = dataset.examples
        self.num_examples = len(self.examples)
        # initialize random weights
        self.w = random_weights(
            min_value=-0.5, max_value=0.5, num_weights=len(self.idx_i) + 1
        )
        # learning loop
        self.learn(learning_rate, epochs)

    def learn(self, learning_rate, epochs):
        """learning loop"""

        def loss(example, w, idx_i, idx_t, h):
            """error: difference between estimation and true value"""
            raise NotImplementedError

        def update(w, learning_rate, err, h, X_col, num_examples):
            """update weights"""
            for i in range(len(w)):
                buffer = [x * y for x, y in zip(err, h)]
                w[i] = w[i] - learning_rate * (np.dot(buffer, X_col[i]) / num_examples)

        def homogeneous(num_examples):
            """build homogeneous coordinates"""
            raise NotImplementedError

        X_col = homogeneous(self.num_examples)
        for epoch in range(epochs):
            err = []
            h = []
            # pass over all examples
            for example in self.examples:
                err.append(loss(example, self.w, self.idx_i, self.idx_t, h))

            # update weights
            update(self.w, learning_rate, err, h, X_col, self.num_examples)

    def predict(self, x):
        """make prediction"""
        return int(np.dot(self.w, [1] + x))


if __name__ == "__main__":
    iris = DataSet(name="iris")
    iris.classes_to_numbers()

    tests = [
        ([5, 3, 1, 0.1], 0),
        ([5, 3.5, 1, 0], 0),
        ([6, 3, 4, 1.1], 1),
        ([6, 2, 3.5, 1], 1),
        ([7.5, 4, 6, 2], 2),
        ([7, 3, 6, 2.5], 2),
    ]

    print(f"===================\nperceptron:")
    perceptron = PerceptionLinearLearner(iris)
    g = grade_learner(perceptron, tests)
    print(f"  learner grade: {g}")
    # assert g > 1. / 2
    e = err_ratio(perceptron, iris)
    print(f"  error ration: {e}")
    # assert e < 0.4

    print(f"===================\nlogistic:")
    logisticer = LogisticLinearLeaner(iris)
    g = grade_learner(logisticer, tests)
    print(f"  learner grade: {g}")
    # assert g > 1. / 2
    e = err_ratio(logisticer, iris)
    print(f"  error ration: {e}")
    # assert e < 0.4
