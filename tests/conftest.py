import numpy as np

def get_dummy_training_data(inputSize, outputSize, count):
    """
    creates a list of randomized (dummy) training data (X, Y)
    of the desired dimentions.
    """
    X = []
    Y = []

    for _ in range(count):
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
        X.append(np.random.rand(inputSize, 1))  # column vector e.g. np.array([[1], [2], [3], [4]])
        Y.append(np.random.rand(outputSize, 1)) # column vector e.g. np.array([[0.0], [1.0], [0.0]])
    return X, Y
