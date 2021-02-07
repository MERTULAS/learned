from time import time


class GradientDescent:

    def __init__(self, data, intercept=0, slope=1, learning_rate=0.00001):
        self.data = data
        self.weights = self.data[data.keys()[0]]
        self.heights = self.data[data.keys()[1]]
        self.intercept = intercept
        self.slope = slope
        self.learning_rate = learning_rate
        self.minimize_val = 1

    def derivatives_sum_of_squared_residuals(self):
        derivative_intercept = 0
        derivative_slope = 0

        for i in range(len(self.data)):
            prediction = self.intercept + self.slope * self.weights[i]
            derivative_intercept += -2 * (self.heights[i] - prediction)
            derivative_slope += -2 * self.weights[i] * (self.heights[i] - prediction)

        step_size_intercept = derivative_intercept * self.learning_rate
        self.intercept = self.intercept - step_size_intercept

        step_size_slope = derivative_slope * self.learning_rate
        self.slope = self.slope - step_size_slope

        return derivative_intercept, derivative_slope

    @staticmethod
    def success_rate(weights, heights, intercept, slope):
        sum_of_squared_residuals = 0
        for i in range(weights.index.start, weights.index.stop):
            prediction = intercept + slope * weights[i]
            sum_of_squared_residuals += (heights[i] - prediction) ** 2
        return sum_of_squared_residuals

    @staticmethod
    def r2_squared(first_ssr, success_rate):
        return (first_ssr - success_rate) / first_ssr

    def optimizer(self, number_of_steps=False):
        intercept_init = self.heights.sum() / len(self.heights)
        first_ssr = self.success_rate(self.weights, self.heights, intercept_init, 0)
        time_start = time()
        if number_of_steps:
            i = 1
            while i <= number_of_steps:
                intercept, slope = self.derivatives_sum_of_squared_residuals()
                print("Epoch:{}______Derivative_intercept:{}, Derivative_slope:{}".format(i, intercept, slope))
                print("___Intercept:{}, Slope:{}".format(self.intercept, self.slope))
                i += 1
        else:
            i = 1
            while True:
                self.minimize_val, slope = self.derivatives_sum_of_squared_residuals()
                print("Epoch:{}______Derivative_intercept:{}, Derivative_slope:{}".format(i, self.minimize_val, slope))
                print("___Intercept:{}, Slope:{}".format(self.intercept, self.slope))
                if 0.0001 > self.minimize_val > -0.0001:
                    break
                i += 1
        time_stop = time()
        benchmark = time_stop - time_start
        success_rate = self.success_rate(self.weights, self.heights, self.intercept, self.slope)
        print("First SSR:{}".format(first_ssr))
        print("Last SSR:{}".format(success_rate))
        print("R-Squared:%{}".format(100 * self.r2_squared(first_ssr, success_rate)))
        print("Completed in {} seconds".format(benchmark))

    def get_parameters(self):
        return self.intercept, self.slope

    def test(self, data):
        try:
            weight = data[data.keys()[0]]
            height = data[data.keys()[1]]
        except TypeError:
            weight = data[0]
            height = data[1]
        intercept_init = height.sum() / len(height)
        first_ssr = self.success_rate(weight, height, intercept_init, 0)
        last_ssr = self.success_rate(weight, height, self.intercept, self.slope)
        r2_squared = self.r2_squared(first_ssr, last_ssr)
        print("Test Score: %{}".format(100 * r2_squared))
