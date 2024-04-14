import numpy as np

def L2_loss(Y_true, Y_hat):
    return np.mean(np.square(Y_true - Y_hat))

def L1_loss(Y_true, Y_hat):
    return np.mean(abs(Y_true - Y_hat))

def categorical_cross_entropy(Y_true, Y_hat):
    return np.mean(-np.sum(Y_true * np.log(Y_hat + 10**-100)))
