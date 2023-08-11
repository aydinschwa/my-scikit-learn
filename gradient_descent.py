import numpy as np
import matplotlib.pyplot as plt


# gradient descent for two variable linear regression


class GDSimpleRegression:
    def __init__(self, n_epochs=100000, learning_rate=.0001):
        # choose random initialization for the two parameters
        self.m = np.random.randint(1, 11)
        self.b = np.random.randint(1, 11) 
        self.learning_rate = learning_rate 
        self.n_epochs = n_epochs 
        self.prev_loss = 99999999
        
    def loss(self, y, y_pred):
        return np.mean(np.square(y - y_pred))

    # GD update equations for y = mX + b
    def update_params(self, X, y, predicted):
        n = len(y)
        m_gradient = (-2/n)*np.sum((y - predicted)*X)

        b_gradient = (-2/n)*np.sum(y - predicted)

        self.m -= self.learning_rate*m_gradient
        self.b -= self.learning_rate*b_gradient
    
    
    def fit(self, X, y):
        for _ in range(self.n_epochs):
            predicted = self.predict(X)
            self.update_params(X, y, predicted)

            loss = self.loss(y, predicted)
            if abs(self.prev_loss - loss) < .001:
                break

            self.prev_loss = loss


    def predict(self, X):
        return self.m*X + self.b 


# create a datset
n = 100
X = np.arange(100)
y = X + 50 + np.random.normal(loc=10, scale=15, size=n)
reg = GDSimpleRegression()
reg.fit(X, y)

plt.scatter(X, y)
y_line = reg.m*X + reg.b
plt.plot(X, y_line, color="red")
plt.show()


