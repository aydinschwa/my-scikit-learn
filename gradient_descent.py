import numpy as np
import matplotlib.pyplot as plt


class GDSimpleRegression:
    def __init__(self, n_epochs=100000, learning_rate=.0001):
        self.learning_rate = learning_rate 
        self.n_epochs = n_epochs 
        self.prev_loss = 99999999
        
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
        for _ in range(self.n_epochs):
            predicted = self.predict(X)
            self.update_params(X, y, predicted)

            loss = self.loss(y, predicted)
            if abs(self.prev_loss - loss) < .001:
                break

            self.prev_loss = loss


    def predict(self, X):
        return np.dot(X, self.w) + self.b 


# create a datset
n = 100
X = np.arange(100)
y = X + 50 + np.random.normal(loc=10, scale=15, size=n)
reg = GDSimpleRegression()
X = np.array(X).reshape(len(X), -1)
reg.fit(X, y)

plt.scatter(X, y)
y_line = reg.w*X + reg.b
plt.plot(X, y_line, color="red")
plt.show()


from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X_iris = iris.data[:, [0, 1, 3]] # Excluding petal length
y_iris = iris.data[:, 2]  # Petal length

# Initialize and fit the regressor
reg_iris = GDSimpleRegression(n_epochs=100000, learning_rate=0.0001)
reg_iris.fit(X_iris, y_iris)

# Predict the petal length
y_pred_iris = reg_iris.predict(X_iris)

# Print the coefficients
print("Coefficients (w):", reg_iris.w)
print("Intercept (b):", reg_iris.b)

