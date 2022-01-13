import numpy as np


def softmax(x, t=1):
    """"
    Applies the softmax temperature on the input x, using the temperature t
    """
    max_x = np.max(x)
    exp_x = np.exp((x - max_x) / t)
    sum_exp = np.sum(exp_x)
    soft_max = exp_x / sum_exp
    return soft_max
