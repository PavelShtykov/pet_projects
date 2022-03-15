import numpy as np


def RMSE(y_preds, y_true):
    return np.mean((y_preds - y_true) ** 2) ** 0.5


def MSE(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)
