import numpy as np
import json
import os
from ..metrics.metrics import accuracy_calc


class Sequential:
    __slots__ = ["NN_structure", "reverse_NN_structure", "x", "y", "learning_rate", "iteration", "cost_list",
                 "loss_func", "loss_func_derivative", "accuracy_list"]

    def __init__(self, x, y, learning_rate=0.01, iteration=1000, loss="binary_cross_entropy"):
        # self.x = np.append(np.ones((1, x.shape[1])), x, axis=0)
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.NN_structure = []
        self.reverse_NN_structure = []
        self.cost_list = []
        self.accuracy_list = []
        loss_functions_list = {"binary_cross_entropy": (self.__bce, self.__d_bce),
                               "cross_entropy": (self.__ce, self.__d_ce),
                               "mean_square_error": (self.__mse, self.__d_mse),
                               "mean_absolute_error": (self.__mae, self.__d_mae)}
        self.loss_func = loss_functions_list[loss][0]
        self.loss_func_derivative = loss_functions_list[loss][1]

    def __layers_weights_initializer(self, x_shape_dim_zero):
        self.NN_structure[0].initializer(x_shape_dim_zero)
        for layer_num in range(1, len(self.NN_structure)):
            self.NN_structure[layer_num].initializer(self.NN_structure[layer_num - 1].neurons)

    def __recursion_forward(self, step, out):
        if step == len(self.NN_structure):
            return out
        else:
            """print(self.NN_structure[step].weights)
            print("\n")"""
            return self.__recursion_forward(step + 1,
                                            self.NN_structure[step].forward_propagation(out))

    def __recursion_backward(self, step, backward_derivatives):
        if step == len(self.reverse_NN_structure):
            pass
        else:
            return self.__recursion_backward(step + 1,
                                             self.reverse_NN_structure[step].backward_propagation(backward_derivatives,
                                                                                                  self.learning_rate))

    def __bce(self, out):
        return self.y * np.log(out) + (1 - self.y) * np.log(1 - out)

    def __d_bce(self, out):
        return -1 * ((self.y - 1.) / (1. - out) + self.y / out)

    def __ce(self, out):
        return self.y * np.log(out)

    def __d_ce(self, out):
        return -1 * (self.y / out)

    def __mse(self, out):
        return -(self.y - out) ** 2

    def __d_mse(self, out):
        return -2 * (self.y - out)

    def __mae(self, out):
        return -np.abs(self.y - out)

    def __d_mae(self, out):
        out[out > self.y] = 1
        out[out < self.y] = -1
        return out

    @staticmethod
    @accuracy_calc
    def __acc(**kwargs):
        pass

    def add(self, layer):
        self.NN_structure.append(layer)

    def train(self):
        self.reverse_NN_structure = self.NN_structure[::-1]
        self.__layers_weights_initializer(self.x.shape[0])
        self.cost_list = []
        softmax_check = 1 if self.NN_structure[-1].activation == "softmax" else 0
        classifier_check = 1 if softmax_check or (self.reverse_NN_structure[0].neurons == 1 and
                                                  self.reverse_NN_structure[0].activation == "sigmoid"
                                                  and len(set(self.y[0])) == 2 and 1 in set(self.y[0])
                                                  and 0 in set(self.y[0])) else 0
        acc_type = "categorical" if classifier_check else "regression"
        for i in range(self.iteration):
            out = self.__recursion_forward(0, self.x)
            loss = self.loss_func(out)
            cost = -loss.sum() / self.y.shape[1]
            self.cost_list.append(cost)
            acc = self.__acc(self.y, out, acc_type) * 100
            self.accuracy_list.append(acc)
            print("\r", f"Epoch:{i + 1}/{self.iteration}......Cost:{cost}......Accuracy: %{acc}", end="")
            if softmax_check:
                d_error = out - self.y
            else:
                d_error = self.loss_func_derivative(out)
            self.__recursion_backward(0, d_error)
        print("\n\nTrain Acc: % " + f"{self.__acc(self.y, self.predict(self.x), acc_type) * 100}")

    def test(self, x, y):
        print("Test Acc: % " + f"{self.__acc(y, self.predict(x)) * 100}")

    def predict(self, x):
        return self.__recursion_forward(0, x)

    def save_model(self, file_name):
        if file_name in os.listdir():
            os.rmdir(file_name)
        os.mkdir(file_name)
        model_functions = {}
        for index, layer in enumerate(self.NN_structure):
            model_functions[index] = layer.activation
            np.save(file_name + f"/weights-{index}.npy", layer.weights)
        with open(file_name + "/activations.json", 'w') as f:
            json.dump(model_functions, f)


class DNNModel:
    __slots__ = ["model_funcs", "weights", "activation_func_list", "model_folder"]

    def __init__(self, model_folder):
        self.model_folder = model_folder
        with open(model_folder + "/activations.json", "r") as f:
            self.model_funcs = json.load(f)
        self.weights = os.listdir(self.model_folder)[1:]
        self.weights.sort()
        self.activation_func_list = {"tanh": self.__tanh, "relu": self.__relu, "sigmoid": self.__sigmoid,
                                     "softmax": self.__softmax, "leaky_relu": self.__leaky_relu}

    def __forward_propagation(self, x):
        out = x
        for index, weight in enumerate(self.weights):
            out = np.append(np.ones((1, out.shape[1])), out, axis=0)
            out = self.activation_func_list[self.model_funcs[str(index)]](
                np.dot(np.load(self.model_folder + "/" + weight), out))
        return out

    @staticmethod
    def __tanh(z):
        return np.tanh(z)

    @staticmethod
    def __sigmoid(z):
        return 1. / (1 + np.exp(-z))

    @staticmethod
    def __relu(z):
        return z * (z > 0)

    @staticmethod
    def __softmax(z):
        exp = np.exp(z - np.max(z))
        return exp / exp.sum(axis=0)

    @staticmethod
    def __leaky_relu(z):
        pass

    def predict(self, x):
        return self.__forward_propagation(x)
