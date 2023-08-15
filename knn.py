import numpy as np
import matplotlib.pyplot as plt
from error_funcs import rmse


class KNNRegression:

    def __init__(self, k=5) -> None:
        self.k = k
        self.X = None 
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def euclidean_distance(self, vec1, vec2):
        return np.sqrt(np.sum(np.square(vec1 - vec2)))

    def predict(self, X):
        out = []
        for row in X:
            out.append(self.predict_obs(row))
        return np.array(out)

    def predict_obs(self, X):
        neighbors = np.zeros(len(self.X))
        for obs_idx in range(len(self.X)):
            obs = self.X[obs_idx, :]
            distance = self.euclidean_distance(obs, X)
            neighbors[obs_idx] = distance
        nearest_neighbor_indices = np.argsort(neighbors)[:self.k]
        nearest_neighbors = self.y[nearest_neighbor_indices]
        return np.mean(nearest_neighbors)

            



if __name__ == "__main__":
    n = 100
    X = np.arange(100)
    y = X + 50 + np.random.normal(loc=10, scale=15, size=n)
    X = np.array(X).reshape(len(X), -1)
    y = np.array(y).reshape(len(y), -1)

    reg = KNNRegression()
    reg.fit(X, y)

    sample = [16, 76]
    out = reg.predict(sample)
    
    plt.scatter(X, y)
    plt.scatter(sample, out, color="red")
    plt.show()

    from sklearn.datasets import load_iris
    iris = load_iris()
    X_iris = iris.data[:, [0, 1, 3]] # Excluding petal length
    y_iris = iris.data[:, 2]  # Petal length

    # Initialize and fit the regressor
    reg_iris = KNNRegression() 
    reg_iris.fit(X_iris, y_iris)

    # Predict the petal length
    y_pred_iris = reg_iris.predict(X_iris)
    print(rmse(y_pred_iris, y_iris))