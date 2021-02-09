from time import time
import numpy as np


class GradientDescent:
    __slots__ = ["data", "slope", "learning_rate", "weights", "heights", "beta", "data_x"]

    def __init__(self, data, slope=1, learning_rate=0.00001):
        self.data = data
        self.data_x = np.array(self.data.iloc[:, :-1].values)
        self.weights = np.append(np.ones((self.data_x.shape[0], 1)),
                                 self.data_x,
                                 axis=1)
        self.heights = np.array(self.data.iloc[:, -1:].values)
        self.slope = slope
        self.beta = np.append(np.ones((1, 1)) * self.heights.sum() / len(self.heights),
                              np.ones((self.data_x.shape[1], 1)) * self.slope,
                              axis=0)
        self.learning_rate = learning_rate

    def derivatives_sum_of_squared_residuals(self):
        prediction = np.dot(self.weights, self.beta)
        pred_gap = prediction - self.heights
        derivative_intercept = 2 * np.dot(self.weights.T[0],  pred_gap)
        self.beta[0] = self.beta[0] - derivative_intercept * self.learning_rate / len(self.heights)
        self.beta[1:] = self.beta[1:] - (2 * (np.dot(self.weights.T[1:],  pred_gap)) * self.learning_rate / len(self.heights))
        return derivative_intercept

    @staticmethod
    def success_rate(weights, heights, beta):
        prediction = np.dot(weights, beta)
        pred_gap = heights - prediction
        sum_of_squared_residuals = (pred_gap ** 2).sum()
        return sum_of_squared_residuals

    @staticmethod
    def r2_squared(first_ssr, success_rate):
        return (first_ssr - success_rate) / first_ssr

    @property
    def first_ssr(self):
        intercept_init = self.heights.sum() / len(self.heights)
        first_beta = np.append(np.ones((1, 1)) * intercept_init,
                               np.ones((self.data_x.shape[1], 1)) * self.slope,
                               axis=0)
        first_ssr = self.success_rate(self.weights, self.heights, first_beta)
        print(first_ssr)
        return first_ssr

    def optimizer(self, number_of_steps=False):
        first_ssr = self.first_ssr
        time_start = time()
        if number_of_steps:
            i = 1
            while i <= number_of_steps:
                self.derivatives_sum_of_squared_residuals()
                print("Epoch:{}".format(i))
                i += 1
        else:
            i = 1
            while 1:
                intercept_derivative = self.derivatives_sum_of_squared_residuals()
                if self.beta[0][0] != self.beta[0][0]:
                    raise Exception("Learning rate should be chosen lower!")
                
                if 0.000001 > abs(intercept_derivative):
                    break
                i += 1
        time_stop = time()
        benchmark = round(time_stop - time_start, 2)
        print("Completed in {} seconds".format(benchmark))
        success_rate = self.success_rate(self.weights, self.heights, self.beta)
        print("R-Squared:%{}".format(100 * self.r2_squared(first_ssr, success_rate)))

    def get_parameters(self):
        return self.beta[0][0], self.beta[1:]

    def test(self, data):
        """if self.intercept != self.intercept or self.slope != self.slope:
            raise Exception("This model not trained!")"""

        try:
            weight_x = data.iloc[:, :-1].values
            weight = np.append(np.ones((weight_x.shape[0], 1)),
                               weight_x,
                               axis=1)
            height = data.iloc[:, -1:].values
        except TypeError:
            weight = data[0]
            height = data[1]
        first_ssr = self.first_ssr
        print(first_ssr)
        last_ssr = self.success_rate(weight, height, self.beta)
        r2_squared = self.r2_squared(first_ssr, last_ssr)
        print("Test Score: %{}".format(100 * r2_squared))
