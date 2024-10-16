import numpy as np


class LinearRegression:
    def solve(self, lr, nepoch):
        raise NotImplementedError


class LinearRegressionLS(LinearRegression):
    """
    solve linear regression problem via least squares
    """

    X: np.array
    Y: np.array
    w: np.array

    def __init__(self, ylist):
        num_data = len(ylist)
        self.X = self.homogeneous([x for x in range(num_data)])
        self.Y = np.array(ylist).reshape(num_data, 1)
        self.w = np.random.rand(2)

    def homogeneous(self, xlist):
        """build homogeneous coordinates"""
        raise NotImplementedError

    def linout(self, xlist):
        """linear output for given data"""
        raise NotImplementedError

    def loss_sq(self, X, Y):
        """loss function: (half) sum of square errors"""
        raise NotImplementedError

    def solve(self, lr, nepoch):
        """form normal equation"""
        XtX = np.dot(self.X.T, self.X)
        XtY = np.dot(self.X.T, self.Y)
        # print(XtX.shape, XtY.shape)
        self.w = np.dot(np.linalg.inv(XtX), XtY).flatten()
        # print(self.w.shape, self.w)


class LinearRegressionGD1(LinearRegression):
    """
    solve linear regression problem via gradient descent,
    using single weight vector
    """

    X: np.array
    Y: np.array
    w: np.array

    def __init__(self, ylist):
        num_data = len(ylist)
        self.X = self.homogeneous([x for x in range(num_data)])
        self.Y = np.array(ylist).reshape(num_data, 1)
        self.w = np.random.rand(2)

    def homogeneous(self, xlist):
        """build homogeneous coordinates"""
        raise NotImplementedError

    def linout(self, xlist):
        """linear output for given data"""
        raise NotImplementedError

    def loss_sq(self, X, Y):
        """loss function: (half) sum of square errors"""
        raise NotImplementedError

    def gd(self, lr):
        """gradient descent update"""

        def gradient(Y_hat, Y, X):
            return np.sum((Y_hat - Y) * X, axis=0)

        Y_hat = self.linout(self.X)
        # print(Y_hat)
        grad = gradient(Y_hat, self.Y, self.X)
        self.w -= lr * grad

    def solve(self, lr, nepoch):
        """iterative solver"""
        for epoch in range(num_epochs):
            self.gd(lr)
            print(f"epoch {epoch + 1}, loss {self.loss_sq(self.X, self.Y)}")


class LinearRegressionGD2(LinearRegression):
    X: np.array
    Y: np.array
    w: np.array
    b: np.array

    def __init__(self, ylist):
        num_data = len(ylist)
        self.X = np.array([x for x in range(num_data)]).reshape(-1, 1)
        self.Y = np.array(ylist).reshape(num_data, 1)
        self.w = np.random.rand(1)
        self.b = np.min(ylist)

    def linout(self, xlist):
        """linear output for given data"""
        raise NotImplementedError

    def loss_sq(self, X, Y):
        """loss function: (half) sum of square errors"""
        raise NotImplementedError

    def gd(self, lr):
        """gradient descent update"""

        def gradient(Y_hat, Y, X):
            return np.array(
                [np.sum((Y_hat - Y) * X, axis=0), np.sum((Y_hat - Y), axis=0)]
            )

        Y_hat = self.linout(self.X)
        # print(Y_hat)
        grad = gradient(Y_hat, self.Y, self.X)
        self.w -= lr * grad[0]
        self.b -= lr * grad[1]

    def solve(self, lr, nepoch):
        """iterative solver"""
        for epoch in range(num_epochs):
            self.gd(lr)
            print(f"epoch {epoch + 1}, loss {self.loss_sq(self.X, self.Y)}")


if __name__ == "__main__":
    hp = [
        14213,
        13448,
        13870,
        16192,
        16415,
        21501,
        25910,
        24866,
        28981,
        32926,
        36741,
        40974,
    ]
    hp = [x / 10000.0 for x in hp]

    lr = 0.001
    num_epochs = 10

    ls = LinearRegressionLS(hp)
    ls.solve(lr, num_epochs)
    print(f"next prediction (LS): {ls.linout([len(hp)])}")

    gd1 = LinearRegressionGD1(hp)
    gd1.solve(lr, num_epochs)
    print(f"next prediction year (GD1): {gd1.linout([len(hp)])}")

    gd2 = LinearRegressionGD1(hp)
    gd2.solve(lr, num_epochs)
    print(f"next prediction year (GD2): {gd2.linout([len(hp)])}")
