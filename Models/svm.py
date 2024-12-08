import numpy as np


class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epsilon=0.1, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for i in range(n_samples):
                y_pred = np.dot(X[i], self.w) + self.b
                error = y[i] - y_pred

                # Update only if error is outside the epsilon-insensitive zone
                if abs(error) > self.epsilon:
                    # Gradient descent step
                    self.w += self.lr * (error * X[i] - 2 * self.lambda_param * self.w)
                    self.b += self.lr * error

    def predict(self, X):
        return np.dot(X, self.w) + self.b