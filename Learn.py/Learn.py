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

    def success_rate(self):
        sum_of_squared_residuals = 0
        for i in range(len(self.data)):
            prediction = self.intercept + self.slope * self.weights[i]
            sum_of_squared_residuals += (self.heights[i] - prediction) ** 2
        return sum_of_squared_residuals

    def r2_squared(self, first_ssr):
        return (first_ssr - self.success_rate()) / first_ssr

    def optimizer(self, number_of_steps=False):
        first_ssr = self.success_rate()
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
                if 0.01 > self.minimize_val > -0.01:
                    break
                i += 1
        time_stop = time()
        benchmark = time_stop - time_start
        print("First SSR:{}".format(first_ssr))
        print("Last SSR:{}".format(self.success_rate()))
        print("R-Squared:%{}".format(100 * self.r2_squared(first_ssr)))
        print("Completed in {} seconds".format(benchmark))

    def get_parameters(self):
        return self.intercept, self.slope
