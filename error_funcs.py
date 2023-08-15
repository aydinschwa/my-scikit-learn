import numpy as np

# Root Mean Square Error
# https://en.wikipedia.org/wiki/Root-mean-square_deviation
def rmse(predicted: np.array, target: np.array) -> float:
    return np.sqrt(np.mean(np.square(target - predicted)))


# Cross-Entropy Loss
# 