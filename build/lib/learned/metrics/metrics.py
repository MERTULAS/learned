import numpy as np


def confusion_matrix(y_true, y_pred):
    types = "label_encoding" if 1 in y_true.shape else "one_hot_encoding"
    if types == "one_hot_encoding":
        y_true, y_pred = y_true.T, y_pred.T
        zeros = np.zeros((y_true.shape[1], y_pred.shape[1]), dtype="int")
        _max = np.max(y_pred, axis=1)
        for index, (pred_row, true_row) in enumerate(zip(y_pred, y_true)):
            i, j = list(pred_row).index(_max[index]), list(true_row).index(1)
            zeros[i, j] += 1
        return zeros
    if types == "label_encoding":
        y_pred, y_true = np.array(np.round(y_pred), dtype="int"), np.array(y_true, dtype="int")
        y_true, y_pred = y_true[0], y_pred[0]
        set_true = set(y_true)
        zeros = np.zeros((len(set_true), len(set_true)), dtype="int")
        for pred_row, true_row in zip(y_pred, y_true):
            zeros[pred_row][true_row] += 1
        return zeros


def accuracy_calc(function):
    def calculator(y_true, y_pred):
        confusion_m = confusion_matrix(y_true, y_pred)
        identity_matrix = np.eye(confusion_m.shape[0], confusion_m.shape[1])
        return (confusion_m * identity_matrix).sum() / confusion_m.sum()
    return calculator


@accuracy_calc
def accuracy(**kwargs):
    pass
