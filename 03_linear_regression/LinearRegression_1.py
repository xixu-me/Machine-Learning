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
        return np.column_stack((np.ones(len(xlist)), xlist))

    def linout(self, xlist):
        """linear output for given data"""
        X = self.homogeneous(xlist)
        return np.dot(X, self.w)

    def loss_sq(self, X, Y):
        """loss function: (half) sum of square errors"""
        Y_pred = self.linout(X[:, 1])
        return 0.5 * np.sum((Y_pred - Y) ** 2)

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
        self.X = np.array([x for x in range(num_data)]).reshape(
            -1, 1
        )  # Changed this line
        self.Y = np.array(ylist).reshape(num_data, 1)
        self.w = np.random.rand(2, 1)  # Changed this line

    def homogeneous(self, xlist):
        """build homogeneous coordinates"""
        return np.column_stack((np.ones(len(xlist)), xlist))

    def linout(self, xlist):
        """linear output for given data"""
        X = self.homogeneous(xlist)
        return np.dot(X, self.w)

    def loss_sq(self, X, Y):
        """loss function: (half) sum of square errors"""
        Y_pred = self.linout(X)
        return 0.5 * np.sum((Y_pred - Y) ** 2)

    def gd(self, lr):
        """gradient descent update"""
        X = self.homogeneous(self.X)
        Y_hat = self.linout(self.X)
        grad = np.dot(X.T, (Y_hat - self.Y))
        self.w -= lr * grad

    def solve(self, lr, nepoch):
        """iterative solver"""
        for epoch in range(nepoch):
            self.gd(lr)
            print(f"epoch {epoch + 1}, loss {self.loss_sq(self.X, self.Y)}")


class LinearRegressionGD2(LinearRegression):
    X: np.array
    Y: np.array
    w: np.array
    b: float  # Changed this line

    def __init__(self, ylist):
        num_data = len(ylist)
        self.X = np.array([x for x in range(num_data)]).reshape(-1, 1)
        self.Y = np.array(ylist).reshape(num_data, 1)
        self.w = np.random.rand(1, 1)  # Changed this line
        self.b = np.random.rand()  # Changed this line

    def linout(self, xlist):
        """linear output for given data"""
        return np.dot(xlist, self.w) + self.b

    def loss_sq(self, X, Y):
        """loss function: (half) sum of square errors"""
        Y_pred = self.linout(X)
        return 0.5 * np.sum((Y_pred - Y) ** 2)

    def gd(self, lr):
        """gradient descent update"""
        Y_hat = self.linout(self.X)
        dw = np.dot(self.X.T, (Y_hat - self.Y))
        db = np.sum(Y_hat - self.Y)
        self.w -= lr * dw
        self.b -= lr * db

    def solve(self, lr, nepoch):
        """iterative solver"""
        for epoch in range(nepoch):
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

    gd2 = LinearRegressionGD2(hp)
    gd2.solve(lr, num_epochs)
    print(f"next prediction year (GD2): {gd2.linout([len(hp)])}")

    lr = 0.03
    gd1.solve(lr, num_epochs)
    print(f"next prediction year (GD1, learning rate 0.03): {gd1.linout([len(hp)])}")
