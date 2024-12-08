import numpy as np

class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # Ensure X and y are numpy arrays
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Ensure binary labels are 1 and -1
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("SVM is designed for binary classification")
        
        y = np.where(y == unique_labels[0], -1, 1)
        
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condition for misclassification
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if not condition:
                    # Update weights
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y[idx] * x_i)
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        # Ensure X is a numpy array
        X = np.asarray(X, dtype=float)
        
        # Return binary predictions (0 or 1)
        linear_model = np.dot(X, self.w) - self.b
        return np.where(linear_model >= 0, 1, 0)
    
    def predict_proba(self, X):
        # Ensure X is a numpy array
        X = np.asarray(X, dtype=float)
        
        # Simplified probability estimation using decision function
        linear_model = np.dot(X, self.w) - self.b
        probas = 1 / (1 + np.exp(-linear_model))
        return np.column_stack((1 - probas, probas))