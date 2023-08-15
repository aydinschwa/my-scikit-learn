import numpy as np
import math



class DecisionTreeBase:
    def __init__(self, depth=0, max_depth=15, min_samples=1, max_features=None):
        self.l_child = None
        self.r_child = None
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.depth = depth
        self.max_features = max_features
        self.split_col = None
        self.split_val = None
        self.is_leaf = False
        self.y_vals = []
        
    def find_best_split(self, data, target):
        min_loss = math.inf
        best_col = None
        best_split = None

        cols = range(len(data[0]))
        if self.max_features:
            cols = np.random.choice((cols), size=self.max_features, replace=False)

        for col_idx in cols: 

            for row_idx in range(len(data) - 1):
                lesser_half = target[data[:, col_idx] <= data[row_idx][col_idx]]
                greater_half = target[data[:, col_idx] > data[row_idx][col_idx]]
                current_loss = self.loss(lesser_half, greater_half)
                if current_loss < min_loss:
                    best_col = col_idx
                    best_split = data[row_idx][col_idx]
                    min_loss = current_loss 
                if min_loss == 0:
                    break
            if min_loss == 0:
                break
                
        return best_col, best_split 
    
    def fit(self, X, y):
        split_col, split_val = self.find_best_split(X, y)
        self.split_col = split_col
        self.split_val = split_val

        if not split_val:
            self.is_leaf = True
            self.y_vals = y
            return

        lesser_criteria = X[:, split_col] <= split_val
        greater_criteria = X[:, split_col] > split_val

        if len(y[lesser_criteria]) < self.min_samples or len(y[greater_criteria]) < self.min_samples:
            self.is_leaf = True
            self.y_vals = y
            
        else:
            self.l_child = type(self)(self.depth + 1)
            self.l_child.fit(X[lesser_criteria], y[lesser_criteria])
            self.r_child = type(self)(self.depth + 1)
            self.r_child.fit(X[greater_criteria], y[greater_criteria])
            
    def predict(self, X):
        if self.is_leaf:
            return self.leaf_predict(self.y_vals)
        if X[self.split_col] <= self.split_val:
            return self.l_child.predict(X)
        else:
            return self.r_child.predict(X)

    # to be implemented by child classes
    def loss(self, less_half, greater_half):
        return None

    def leaf_predict(self, y):
        return None
    

class DecisionTreeClassifier(DecisionTreeBase):
    
    def loss(self, less, greater):
        def get_impurity(vals):
            # get count of each unique value in the dataset
            _, counts = np.unique(vals, return_counts=True)

            # gini impurity formula
            impurity = 1 - np.sum((counts / len(vals))**2)
            return impurity

        return (get_impurity(less) + get_impurity(greater)) / 2

    def leaf_predict(self, y):
        return np.bincount(y).argmax()


class DecisionTreeRegressor(DecisionTreeBase):

    def loss(self, less, greater):
        def sum_sq_error(vals):
            return np.sum((vals - np.mean(vals))**2)
        
        return sum_sq_error(less) + sum_sq_error(greater)

    def leaf_predict(self, y):
        return np.mean(y)

