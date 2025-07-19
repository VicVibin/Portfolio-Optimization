import numpy as np
import matplotlib.pyplot as plt


def Backshift(X_t, k):
    return np.array(X_t[: len(X_t) - k])


def stack(X_t, k):
    return np.array([0] * k + list(X_t))


def fractional_differencing(X_t, d):
    weights, Array = np.zeros(len(X_t))
    weights[0] = 1
    for i in range(1, len(X_t)):
        param = -weights[i - 1] * (d - i + 1) / i
        weights[i] = param if abs(param) > 0.01 else 0

    for i in range(len(X_t)):
        Array += weights[i] * stack(Backshift(X_t, i), i)
    return Array
