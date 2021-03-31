import numpy as np


class OneHotEncoder:
    __slots__ = ["data", "values"]

    def __init__(self, data):
        self.data = data
        self.values = None

    def transform(self):
        try:
            self.data = self.data.values.T
        except AttributeError:
            self.data = self.data.T
        try:
            data_shape = self.data.shape[1]
        except IndexError:
            self.data = np.array([self.data])
        temp = list(set(self.data[0]))
        temp.sort()

        try:
            result = np.empty((self.data.shape[1], len(temp)))
        except IndexError:
            result = np.empty((self.data.shape[0], len(temp)))

        the_dict = {}
        for index, element in enumerate(temp):
            zeros = np.zeros((1, len(temp)))
            zeros[0][index] = 1
            the_dict[element] = zeros[0]
        self.values = the_dict
        for i in range(len(self.data.T)):
            result[i] = the_dict[self.data[0].T[i]]
        return result.T


def get_split_data(data, test_percentage=0.33, random_state=0):
    try:
        data = data.iloc[:, :].values
    except AttributeError:
        pass
    np.random.seed(random_state)
    data_len = len(data)
    random_indexes = [i for i in range(data_len)]
    np.random.shuffle(random_indexes)
    data_test = np.array([data[i] for i in random_indexes[:round(data_len * test_percentage)]])
    data_train = np.array([data[i] for i in random_indexes[round(data_len * test_percentage):]])
    return data_train, data_test


def normalizer(data):
    try:
        data = data.values
    except AttributeError:
        pass
    _max = np.max(data)
    return data / _max


def polynomial_features(data, degree=2):

    try:
        data_x = data.values
    except AttributeError:
        data_x = data
    new = np.empty((len(data_x), 0))
    temp = data_x
    len_new = len(data_x[0])
    while degree > 1:
        for i in range(len(data_x[0])):
            for j in range(i, len_new):
                cross = data_x[:, i] * temp[:, j]
                if list(cross) not in map(lambda x: list(x), new.T):
                    new = np.append(new, cross.reshape(len(data_x), 1), axis=1)
        len_new = len(new[0])
        degree -= 1
        temp = new
    new = np.append(data_x, new, axis=1)
    return new
