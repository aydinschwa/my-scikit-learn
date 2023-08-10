import numpy as np
import sklearn.datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error



class GradientBoostRegression:

    def __init__(self, n_estimators=50, learning_rate=.1): 
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate 
        self.average = None
        self.trees = []


    def fit(self, X, y):
        # initially predict output as average of all output observations
        prediction = np.mean(y)
        self.average = prediction
        # create pseudo residuals
        resid = y - prediction

        for _ in range(self.n_estimators):
            
            # fit decision tree to pseudo residuals 
            tree = DecisionTreeRegressor()
            tree.fit(X, resid)
            self.trees.append(tree)

            adj_prediction = self.average
            for tree in self.trees:
                pred = tree.predict(X)
                adj_prediction += self.learning_rate*pred

            resid = y - adj_prediction 


    def predict(self, X):
        preds = []
        for row in X:
            pred = self.average + self.learning_rate*np.sum([tree.predict([row]) for tree in self.trees])
            preds.append(round(pred, 1))
        return preds 
    


class GBRegression:

    def __init__(self) -> None:
        self.n_estimators = 50
        self.learning_rate = .1
        self.trees = []

    def fit(self, X, y):

        # loss function is .5(obs - predicted)^2, pd wrt predicted is -(obs - predicted)
        # so if we want to step in the direction of minimum we use (obs - predicted)

        # start with a constant value that minimizes sum of residuals, which ends up being the average
        average = np.mean(y)
        self.average = average
        for _ in range(self.n_estimators):
            
            # subtract from residuals to get pseudo residuals aka negative gradients
            psr = y - average

            # now we want to build trees that estimate psr so that when we aggregate their predictions
            # with the original prediction, residuals are minimized
            tree = DecisionTreeRegressor()
            tree.fit(X, psr)

            average = average + self.learning_rate*tree.predict(X)

            self.trees.append(tree)

    def predict(self, X):
        preds = []
        for row in X:
            pred = self.average + self.learning_rate*np.sum([tree.predict([row]) for tree in self.trees])
            preds.append(round(pred, 1))
        return preds 



diabetes_df = sklearn.datasets.load_diabetes()
data = diabetes_df["data"]
target = diabetes_df["target"]
gbr = GBRegression()
gbr.fit(data, target)
y_pred = gbr.predict(data)

print(mean_squared_error(y_pred, target))



gbr = GradientBoostRegression()
gbr.fit(data, target)
y_pred = gbr.predict(data)

print(mean_squared_error(y_pred, target))

# rmse(y_pred, target)



    