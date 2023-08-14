import numpy as np

def rmse(predicted: np.array, target: np.array) -> float:
    return np.sqrt(np.mean(np.square(target - predicted)))