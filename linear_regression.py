import numpy as np
import copy
import matplotlib.pyplot as plt
from error_funcs import rmse


class GDRegression:
    def __init__(self, n_epochs=100000, learning_rate=.0001, animate=False):
        self.learning_rate = learning_rate 
        self.n_epochs = n_epochs 
        self.prev_loss = 99999999
        self.animate = animate
        self.w_history = []
        self.b_history = []
        
    def loss(self, y, y_pred):
        return np.mean(np.square(y - y_pred))

    # GD update equations for y = mX + b
    def update_params(self, X, y, predicted):
        n = len(y)

        w_gradient = (-2/n)*np.dot(X.T, y - predicted)

        b_gradient = (-2/n)*np.sum(y - predicted)

        self.w -= self.learning_rate*w_gradient
        self.b -= self.learning_rate*b_gradient
    
    
    def fit(self, X, y):
        # initialize parameter values
        self.w = np.zeros(len(X[0])) 
        self.b = 0 
        for i in range(self.n_epochs):
            predicted = self.predict(X)
            self.update_params(X, y, predicted)

            loss = self.loss(y, predicted)
            if abs(self.prev_loss - loss) < .001:
                break

            if self.animate: 
                # Store the weights and bias every 10 epochs for animation
                if i % 1500 == 0 or (i < 5):
                    self.w_history.append(self.w.copy())
                    self.b_history.append(self.b)

            self.prev_loss = loss


    def predict(self, X):
        return np.dot(X, self.w) + self.b 
    

class OLSRegression:
    def __init__(self):
        self.parameters = None

    def fit(self, X, y):
        X_copy = copy.deepcopy(X)
        constants = np.ones((X.shape[0], 1))
        X_copy = np.concatenate((constants, X_copy), 1) 
        parameters = np.dot(np.dot(np.linalg.inv(np.dot(X_copy.T, X_copy)), X_copy.T), y)
        self.b = parameters[0]
        self.w = parameters[1:]

    def predict(self, y):
        return np.dot(y, self.w) + self.b


if __name__ == "__main__":
    # create a datset
    n = 100
    X = np.arange(100)
    y = X + 50 + np.random.normal(loc=10, scale=15, size=n)
    reg = OLSRegression() 
    X = np.array(X).reshape(len(X), -1)
    reg.fit(X, y)

    plt.scatter(X, y)
    y_line = reg.w*X + reg.b
    plt.plot(X, y_line, color="red")
    plt.show()


    from sklearn.datasets import load_iris
    iris = load_iris()
    X_iris = iris.data[:, [0, 1, 3]] # Excluding petal length
    y_iris = iris.data[:, 2]  # Petal length

    # Initialize and fit the regressor
    reg_iris = GDRegression(n_epochs=100000, learning_rate=0.0001)
    reg_iris = OLSRegression()
    reg_iris.fit(X_iris, y_iris)

    # Predict the petal length
    y_pred_iris = reg_iris.predict(X_iris)

    # Print the coefficients
    print("Coefficients (w):", reg_iris.w)
    print("Intercept (b):", reg_iris.b)
    print(rmse(y_pred_iris, y_iris))
