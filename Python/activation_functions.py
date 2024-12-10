import numpy as np

def sigmoid(x):
    """
    Sigmoid activation function applied element-wise.
    Parameters:
        x: NumPy array or scalar
    Returns:
        NumPy array or scalar after applying the sigmoid function
    """
    return 1 / (1 + np.exp(-x))