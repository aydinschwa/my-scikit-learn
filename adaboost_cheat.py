import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class AdaBoost():

    def __init__(self, n_estimators=20): 
        self.n_estimators = n_estimators 
        self.trees = []
        self.influence = []

    def fit(self, X, y):

        weights = 1 / len(y) * np.ones((len(y),))
        epsilon = 1e-10 # to prevent div by zero

        for _ in range(self.n_estimators):
            # fit decision tree using sample weights
            tree = DecisionTreeClassifier()
            tree.fit(X, y, sample_weight=weights)

            # predict entire training dataset, get total error
            preds = tree.predict(X)
            error = np.sum(weights[preds != y])

            # calculate influence
            alpha = .5*np.log((1 - error) / (error + epsilon))

            # store tree and influence for prediction later
            self.trees.append(tree)
            self.influence.append(alpha)

            # update weights
            update_direction = np.where(preds == y, -1, 1) # add weight to misclassified obs
            weights *= np.exp(alpha*update_direction)
            weights /= np.sum(weights) # normalize weights

    def predict(self, X):

        def predict_instance(row):
            class_scores = {}
            for tree, influence in zip(self.trees, self.influence):
                pred = tree.predict([row])[0]
                if pred in class_scores:
                    class_scores[pred] += influence
                else:
                    class_scores[pred] = influence

            return max(class_scores, key=class_scores.get)
        
        return np.array([predict_instance(row) for row in X])


# load data 
iris_df = sklearn.datasets.load_iris()
data = iris_df["data"]
target = iris_df["target"]
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33, random_state=42)

boost = AdaBoost()
boost.fit(X_train, y_train)
preds = boost.predict(X_test) 
print(np.sum(np.where(preds == y_test, 1, 0)) / len(y_test))