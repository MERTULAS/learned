import numpy as np


class Layer:
    __slots__ = ["neurons", "weights_initializer", "activation",
                 "input_layer", "weights", "out", "prev_out", "z",
                 "activation_func", "activation_func_derivative",
                 "weights_initializer_func"]

    def __init__(self, *args, **kwargs):
        try:
            self.neurons = kwargs.get("neurons") if "neurons" in kwargs else args[0]
        except IndexError:
            raise Exception(
                'The number of "neurons" must be entered! >>> Layer(n_neurons, ...) or Layer(neurons=n, ...)')

        self.weights_initializer = kwargs.get("weights_initializer") if "weights_initializer" in kwargs else "uniform"
        self.activation = kwargs.get("activation") if "activation" in kwargs else "tanh"
        self.input_layer = kwargs.get("input_layer") if "input_layer" in kwargs else None
        self.weights = None
        self.out = None
        self.prev_out = None
        self.z = None
        function_list = {"tanh": (self.__tanh, self.__d_tanh),
                         "sigmoid": (self.__sigmoid, self.__d_sigmoid),
                         "relu": (self.__relu, self.__d_relu),
                         "leaky_relu": (self.__leaky_relu, self.__d_leaky_relu),
                         "softmax": (self.__softmax, self.__d_softmax)}

        initializer_list = {"he_uniform": self.__he_uniform,
                            "he_normal": self.__he_normal,
                            "xavier_uniform": self.__xavier_uniform,
                            "xavier_normal": self.__xavier_normal,
                            "uniform": self.__normal_uniform}

        self.weights_initializer_func = initializer_list[self.weights_initializer]
        self.activation_func = function_list[self.activation][0]
        self.activation_func_derivative = function_list[self.activation][1]

    def initializer(self, input_layer_shape):
        weights = self.weights_initializer_func(input_layer_shape)
        self.weights = np.append(np.full((self.neurons, 1), 0.0), weights, axis=1)

    def forward_propagation(self, prev_layer_out):
        self.prev_out = np.append(np.ones((1, prev_layer_out.shape[1])), prev_layer_out, axis=0)
        self.z = np.dot(self.weights, self.prev_out)
        self.out = self.activation_func(self.z)
        return self.out

    def backward_propagation(self, next_layer_derivative, learning_rate):
        step_calculate = np.multiply(next_layer_derivative, self.activation_func_derivative(self.z))
        derivative_to_return = np.dot(self.weights[:, 1:].T, step_calculate)
        derivative_of_weights = np.dot(step_calculate, self.prev_out.T) / self.prev_out.shape[1]
        self.weights = self.weights - learning_rate * derivative_of_weights
        return derivative_to_return

    def __he_uniform(self, size):
        return np.random.randn(self.neurons, size) * np.sqrt(6. / size)

    def __he_normal(self, size):
        return np.random.randn(self.neurons, size) * np.sqrt(2. / size)

    def __xavier_uniform(self, size):
        return np.random.randn(self.neurons, size) * np.sqrt(6. / (size + self.neurons))

    def __xavier_normal(self, size):
        return np.random.randn(self.neurons, size) * np.sqrt(2. / (size + self.neurons))

    def __normal_uniform(self, size):
        return np.random.randn(self.neurons, size) * .1

    @staticmethod
    def __tanh(z):
        return np.tanh(z)

    def __d_tanh(self, z):
        return 1. - self.__tanh(z) ** 2

    @staticmethod
    def __sigmoid(z):
        return 1. / (1 + np.exp(-z))

    def __d_sigmoid(self, z):
        return self.__sigmoid(z) * (1. - self.__sigmoid(z))

    @staticmethod
    def __relu(z):
        return z * (z > 0)

    @staticmethod
    def __d_relu(z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z

    @staticmethod
    def __softmax(z):
        exp = np.exp(z - np.max(z))
        return exp / exp.sum(axis=0)

    @staticmethod
    def __d_softmax(z):
        return 1

    @staticmethod
    def __leaky_relu(z):
        pass

    @staticmethod
    def __d_leaky_relu(z):
        pass
