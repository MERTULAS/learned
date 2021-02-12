from time import time
import numpy as np


class GradientDescent:
    __slots__ = ["learning_rate", "weights", "heights", "beta", "data_x", "cost"]

    def __init__(self, data, learning_rate=0.00001):
        try:
            self.data_x = np.array(data.iloc[:, :-1].values)
        except AttributeError:
            self.data_x = np.array(data[:, :-1])
        self.weights = np.append(np.ones((self.data_x.shape[0], 1)),
                                 self.data_x,
                                 axis=1)
        try:
            self.heights = np.array(data.iloc[:, -1:].values)
        except AttributeError:
            self.heights = np.array(data[:, -1:])
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

    def test(self, data):
        try:
            weight_x = data.iloc[:, :-1].values
        except AttributeError:
            weight_x = data[:, :-1]
        weight = np.append(np.ones((weight_x.shape[0], 1)),
                           weight_x,
                           axis=1)
        try:
            height = data.iloc[:, -1:].values
        except AttributeError:
            height = data[:, -1:]

        test_data_predict = np.dot(weight, self.beta)
        r2_score = self.r2_score(height, test_data_predict)
        print("Test Score: %{}".format(100 * r2_score))

    def get_parameters(self):
        return self.beta[0][0], self.beta[1:]


class LinReg:
    __slots__ = ["x_train", "x", "y", "theta_init", "theta"]

    def __init__(self, data):
        try:
            self.x_train = np.array(data.iloc[:, :-1].values)
        except AttributeError:
            self.x_train = np.array(data[:, :-1])
        self.x = np.append(np.ones((self.x_train.shape[0], 1)),
                           self.x_train,
                           axis=1)
        try:
            self.y = np.array(data.iloc[:, -1:].values)
        except AttributeError:
            self.y = np.array(data[:, -1:])
        self.theta_init = np.append(np.ones((1, 1)) * (self.y.sum() / len(self.y)),
                                    np.ones((self.x_train.shape[1], 1)) * 0,
                                    axis=0)
        self.theta = None

    def __r2_inside(self, x, y):
        first_ssr = ((y - np.dot(x, self.theta_init)) ** 2).sum()
        last_ssr = ((y - np.dot(x, self.theta)) ** 2).sum()
        return (first_ssr - last_ssr) / first_ssr

    def train(self):
        start_time = time()
        self.theta = np.dot(np.dot(np.linalg.inv(np.dot(self.x.T, self.x)), self.x.T), self.y)
        end_time = time()
        print(f"Completed in {round(end_time - start_time, 2)} seconds.")
        print(f"Training R2-Score: % {self.__r2_inside(self.x, self.y) * 100}")
        print(f"Intercept: {self.theta[0][0]}, Coefficients: {self.theta[1:].reshape(1, len(self.theta) - 1)}")

    def test(self, t_data):
        if self.theta is not None:
            try:
                x_test = np.array(t_data.iloc[:, :-1].values)
            except AttributeError:
                x_test = np.array(t_data[:, :-1])
            x = np.append(np.ones((x_test.shape[0], 1)),
                          x_test,
                          axis=1)
            try:
                y = np.array(t_data.iloc[:, -1:].values)
            except AttributeError:
                y = np.array(t_data[:, -1:])
            self.theta_init[0] = (y.sum() / len(y))
            print(f"Testing R2-Score: % {self.__r2_inside(x, y) * 100}")
        else:
            raise Exception("Model not trained!")

    def predict(self, x):
        if self.theta is not None:
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


class Preprocessing:
    __slots__ = ["data"]

    def __init__(self, data):
        self.data = data

    def get_split_data(self, test_percentage=0.33, randomizer=True):
        if randomizer:
            data_len = len(self.data)
            random_indexes = [i for i in range(data_len)]
            np.random.shuffle(random_indexes)
            data_test = [self.data.iloc[:, :].values[i] for i in random_indexes[:round(data_len * test_percentage)]]
            data_train = [self.data.iloc[:, :].values[i] for i in random_indexes[round(data_len * test_percentage):]]
            data_test = np.array(data_test, dtype="int")
            data_train = np.array(data_train, dtype="int")
            return data_train, data_test
        else:
            pass
