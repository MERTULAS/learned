import numpy as np
from time import time


class GradientDescent:
    __slots__ = ["learning_rate", "weights", "heights", "beta", "data_x", "cost"]

    def __init__(self, data_x, data_y, learning_rate=0.00001):
        try:
            self.data_x = data_x.values
        except AttributeError:
            self.data_x = data_x
        self.weights = np.append(np.ones((self.data_x.shape[0], 1)),
                                 self.data_x,
                                 axis=1)
        try:
            self.heights = data_y.values
        except AttributeError:
            self.heights = data_y
        self.beta = np.append(np.ones((1, 1)) * self.heights.sum() / len(self.heights),
                              np.zeros((self.data_x.shape[1], 1)),
                              axis=0)
        self.learning_rate = learning_rate
        self.cost = []

    def __derivatives_sum_of_squared_residuals(self):
        prediction = np.dot(self.weights, self.beta)
        pred_gap = prediction - self.heights
        self.cost.append((pred_gap ** 2).sum())
        derivative = 2 * np.dot(self.weights.T,  pred_gap)
        self.beta = self.beta - (derivative * self.learning_rate)
        return derivative[0]

    @staticmethod
    def r2_score(y_true, y_pred):
        first_ssr = ((y_true - (y_true.sum() / len(y_true))) ** 2).sum()
        last_ssr = ((y_true - y_pred) ** 2).sum()
        return (first_ssr - last_ssr) / first_ssr

    def optimizer(self, number_of_steps=False):
        time_start = time()
        if number_of_steps:
            i = 1
            while i <= number_of_steps:
                self.__derivatives_sum_of_squared_residuals()
                if self.beta[0][0] != self.beta[0][0]:
                    raise Exception("Learning rate should be chosen lower!")
                print("\r", f"Iteration:{i}", end="")
                i += 1
        else:
            i = 0
            while 1:
                i += 1
                intercept_derivative = self.__derivatives_sum_of_squared_residuals()
                if self.beta[0][0] != self.beta[0][0]:
                    raise Exception("Learning rate should be chosen lower!")
                print("\r", f"Intercept Slope:{intercept_derivative}", end="")
                if i < 2:
                    continue
                if 0.0001 > abs(intercept_derivative):
                    break
        time_stop = time()
        benchmark = round(time_stop - time_start, 2)
        print("\nCompleted in {} seconds".format(benchmark))
        model_output_predict = np.dot(self.weights, self.beta)
        r2_score = self.r2_score(self.heights, model_output_predict)
        print("R-Squared:%{}".format(100 * r2_score))

    def test(self, data_x, data_y):
        try:
            x = data_x.values
        except AttributeError:
            x = data_x
        x_test = np.append(np.ones((x.shape[0], 1)),
                           x,
                           axis=1)
        try:
            height = data_y.values
        except AttributeError:
            height = data_y

        test_data_predict = np.dot(x_test, self.beta)
        r2_score = self.r2_score(height, test_data_predict)
        print("Test Score: %{}".format(100 * r2_score))

    def get_parameters(self):
        return self.beta[0][0], self.beta[1:]

    def predict(self, x):
        input_x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
        return np.dot(input_x, self.beta)


class LinReg:
    __slots__ = ["x_train", "x", "y", "theta_init", "theta"]

    def __init__(self, data_x, data_y):
        try:
            self.x_train = data_x.values
        except AttributeError:
            self.x_train = data_x
        self.x = np.append(np.ones((self.x_train.shape[0], 1)),
                           self.x_train,
                           axis=1)
        try:
            self.y = data_y.values
        except AttributeError:
            self.y = data_y
        self.theta_init = np.append(np.ones((1, 1)) * (self.y.sum() / len(self.y)),
                                    np.ones((self.x_train.shape[1], 1)) * 0,
                                    axis=0)
        self.theta = np.empty((len(self.x), 0))

    def __r2_inside(self, x, y):
        first_ssr = ((y - np.dot(x, self.theta_init)) ** 2).sum()
        last_ssr = ((y - np.dot(x, self.theta)) ** 2).sum()
        return (first_ssr - last_ssr) / first_ssr

    def train(self):
        start_time = time()
        self.theta = np.linalg.lstsq(self.x, self.y, rcond=None)[0]
        end_time = time()
        print(f"Completed in {round(end_time - start_time, 2)} seconds.")
        print(f"Training R2-Score: % {self.__r2_inside(self.x, self.y) * 100}")
        print(f"Intercept: {self.theta[0][0]}, Coefficients: {self.theta[1:].reshape(1, len(self.theta) - 1)}")

    def test(self, test_x, test_y):
        if self.theta.size != 0:
            try:
                x_test = test_x.values
            except AttributeError:
                x_test = test_x
            x = np.append(np.ones((x_test.shape[0], 1)),
                          x_test,
                          axis=1)
            try:
                y = test_y.values
            except AttributeError:
                y = test_y
            self.theta_init[0] = (y.sum() / len(y))
            print(f"Testing R2-Score: % {self.__r2_inside(x, y) * 100}")
        else:
            raise Exception("Model not trained!")

    def predict(self, x):
        if self.theta.size != 0:
            input_x = np.append(np.ones((x.shape[0], 1)), x, axis=1)
            return np.dot(input_x, self.theta)
        else:
            raise Exception("Model not trained!")

    @staticmethod
    def r2_score(y_true, y_predict):
        first_ssr = ((y_true - (y_true.sum() / len(y_true))) ** 2).sum()
        last_ssr = ((y_true - y_predict) ** 2).sum()
        return (first_ssr - last_ssr) / first_ssr

    @property
    def intercept(self):
        if self.theta is not None:
            return self.theta[0][0]
        else:
            raise Exception("Model not trained!")

    @property
    def coefficients(self):
        if self.theta is not None:
            return self.theta[1:].reshape(1, len(self.theta) - 1)
        else:
            raise Exception("Model not trained!")


class LogReg:
    __slots__ = ["x", "y", "learning_rate", "iteration", "weights"]

    def __init__(self, x, y, learning_rate=0.01, iteration=1000):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = np.append(np.full((1, 1), 0.0), np.full((self.x.shape[0], 1), 0.01), axis=0)
        self.x = np.append(np.ones((1, self.x.shape[1])), self.x, axis=0)

    @staticmethod
    def __sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def __forward_propagation(self):
        return np.dot(self.weights.T, self.x)

    def __backward_propagation(self, y):
        return np.dot(self.x, (y - self.y).T) / self.x.shape[1]

    def __inner_predict(self, x):
        z = self.__sigmoid(np.dot(self.weights.T, x))
        prediction = np.zeros((1, x.shape[1]))
        for i in range(z.shape[1]):
            prediction[0, i] = 0 if z[:, i] <= 0.5 else 1
        return prediction

    @staticmethod
    def accuracy(y_true, y_pred):
        return 100 - np.average(np.abs(y_pred - y_true)) * 100

    def train(self):
        cost_list = []
        for i in range(self.iteration):
            out = self.__sigmoid(self.__forward_propagation())
            loss = -self.y * np.log(out) - (1 - out) * np.log(1 - out)
            cost = loss.sum() / self.x.shape[1]
            cost_list.append(cost)
            derivatives = self.__backward_propagation(out)
            self.weights = self.weights - self.learning_rate * derivatives
            print("\r", f"Iteration:{i}", end="")
        print(f"\n%{self.accuracy(self.y, self.__inner_predict(self.x))}")

    def predict(self, x):
        x = np.append(np.ones((1, x.shape[1])), x, axis=0)
        return self.__inner_predict(x)
