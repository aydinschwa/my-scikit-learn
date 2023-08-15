import numpy as np
import math
from dtree import DecisionTreeClassifier, DecisionTreeRegressor
from error_funcs import rmse
import sklearn.datasets



def bootstrap(X):
    bootstrap_indices = np.random.randint(len(X), size=len(X))
    oob_indices = set(range(len(X))).difference(set(bootstrap_indices))
    return bootstrap_indices, np.array(oob_indices)


class RandomForestClassifier:

    def __init__(self, n_estimators=20, max_features=True):
        self.n_estimators = n_estimators
        self.max_features = int(math.sqrt(max_features))
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            bootstrap_indices, oob_indices = bootstrap(X)
            tree = DecisionTreeClassifier(max_features=self.max_features)
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            self.trees.append(tree)

    def predict(self, X):
        return np.argmax(np.bincount([tree.predict(X) for tree in self.trees]))


class RandomForestRegressor:

    def __init__(self, n_estimators=20, max_features=False):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_estimators):
            bootstrap_indices, oob_indices = bootstrap(X)
            tree = DecisionTreeRegressor()
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            self.trees.append(tree)

    def predict(self, X):
        predictions= []
        for sample in X:
            predictions.append(np.mean([tree.predict([sample]) for tree in self.trees]))
        return np.array(predictions) 


if __name__ == "__main__":
    diabetes_df = sklearn.datasets.load_diabetes()
    data = diabetes_df["data"]
    target = diabetes_df["target"]
    rf = RandomForestRegressor()
    rf.fit(data, target)
    y_pred = rf.predict(data)

    rmse(y_pred, target)