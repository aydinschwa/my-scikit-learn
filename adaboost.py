import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

# np.random.seed(124)

class DecisionStump:

    def __init__(self):
        self.split_col = None 
        self.best_split = None
        self.y_low = None
        self.y_high = None

    # 1 - (p_1)^2 - (p_2)^2 - ...
    def gini(self, vals, weights):
        _, counts = np.unique(vals, return_counts=True)
        return 1 - np.sum((counts / np.sum(counts))**2)

    def fit(self, X, y, weights):
        
        min_loss = 1
        split_col = None
        best_split = None
        y_low = None  
        y_high = None  

        for col_idx in range(len(X[0])):
            for row_idx in range(len(X)):
                less_split = y[X[:, col_idx] <= X[row_idx][col_idx]]
                greater_split = y[X[:, col_idx] > X[row_idx][col_idx]]
                loss = (self.gini(less_split) + self.gini(greater_split)) / 2
                if loss < min_loss:
                    min_loss = loss
                    split_col = col_idx
                    best_split = X[row_idx][col_idx]
                    y_low = less_split 
                    y_high = greater_split 

        self.split_col = split_col
        self.best_split = best_split
        self.y_low = np.argmax(np.bincount(y_low))
        self.y_high = np.argmax(np.bincount(y_high))


    def predict(self, X):
        if X[self.split_col] <= self.best_split:
            return self.y_low 
        else:
            return self.y_high 
        

class AdaBoost:

    def __init__(self, n_estimators=20):
        self.n_estimators = n_estimators 
        self.trees = []
        self.tree_influence = []

    def fit(self, X, y):
        epsilon = .00000001 # to avoid div by 0 error
        weights = np.array([1/len(y) for _ in range(len(y))])
        for _ in range(self.n_estimators): 
            # fit a decision stump
            stump = DecisionStump()
            stump.fit(X, y, weights)
            self.trees.append(stump)
            # make predictions on the entire dataset
            preds = np.array([stump.predict(row) for row in X])

            # get total prediction error
            error = np.sum(weights[preds != y])

            # calculate alpha
            alpha = .5*np.log((1 - error) / (error + epsilon))
            bias_direction = np.where(preds != y, 1, -1)
            self.tree_influence.append(alpha)

            # update weights, normalize
            weights *= np.exp(alpha*bias_direction)
            weights /= np.sum(weights)

    def predict(self, X):
        class_scores = {}
        for tree, influence in zip(self.trees, self.tree_influence):
            pred = tree.predict(X)
            if pred in class_scores:
                class_scores[pred] += influence
            else:
                class_scores[pred] = influence

        return max(class_scores, key=class_scores.get)


iris_df = sklearn.datasets.load_iris()
data = iris_df["data"]
target = iris_df["target"]

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)
stump = DecisionStump()
stump.fit(X_train, y_train)
print(stump.predict(X_test[0]))

boost = AdaBoost()
boost.fit(X_train, y_train)
preds = np.array([boost.predict(row) for row in X_test])
print(np.sum(np.where(preds == y_test, 1, 0)) / len(y_test))


preds = np.array([boost.predict(row) for row in X_train])
print(np.sum(np.where(preds == y_train, 1, 0)) / len(y_train))

# from sklearn.ensemble import AdaBoostClassifier
# boost = AdaBoostClassifier()
# boost.fit(X_train, y_train)
# preds = boost.predict(X_test)
# print(np.sum(np.where(preds == y_test, 1, 0)) / len(y_test))

