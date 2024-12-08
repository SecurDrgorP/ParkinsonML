from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

class CatBoostScratch:
    def __init__(self, n_estimators=500, learning_rate=0.1, max_depth=3, l2_reg=0.1, early_stopping_rounds=50):
        """
        Initialize CatBoost parameters.
        
        Parameters:
        - n_estimators: Total number of trees (boosting iterations).
        - learning_rate: Learning rate to update predictions.
        - max_depth: Maximum depth of trees.
        - l2_reg: L2 regularization to reduce overfitting.
        - early_stopping_rounds: Number of iterations without improvement before stopping training.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.early_stopping_rounds = early_stopping_rounds
        self.trees = []
        self.base_prediction = None
        self.feature_importances_ = None

    def _log_loss_gradient(self, y_true, y_pred):
        """
        Calculate the gradient of log-loss with respect to predictions.
        
        Parameters:
        - y_true: True labels.
        - y_pred: Current predictions (logits).
        
        Returns:
        - Log-loss gradient.
        """
        sigmoid = 1 / (1 + np.exp(-y_pred))  # Apply sigmoid function
        return y_true - sigmoid

    def _check_balance(self, y):
        """
        Check the class balance in labels y.
        """
        pos_ratio = np.mean(y)
        if pos_ratio < 0.1 or pos_ratio > 0.9:
            print("Warning: Imbalanced dataset detected. Consider balancing your data.")

    def fit(self, X, y, validation_data=None):
        """
        Train the CatBoost model using boosting logic.
        
        Parameters:
        - X: Input features.
        - y: Target labels.
        - validation_data: Tuple (X_val, y_val) for validation and early stopping.
        """
        # Encode categorical features if necessary
        if isinstance(X, np.ndarray) and X.dtype.kind in 'OSU':  # Check if X contains strings or categories
            encoder = OrdinalEncoder()
            X = encoder.fit_transform(X)

        # Check class balance
        self._check_balance(y)

        # Initialize base predictions (log-odds for binary classification)
        pos_ratio = np.mean(y)
        self.base_prediction = np.log(pos_ratio / (1 - pos_ratio))
        predictions = np.full(len(y), self.base_prediction)

        # Initialize feature importance tracking
        self.feature_importances_ = np.zeros(X.shape[1])

        # Variables for early stopping
        best_loss = float('inf')
        no_improvement_count = 0

        for iteration in range(self.n_estimators):
            # Calculate residuals (gradients) with L2 regularization
            residuals = self._log_loss_gradient(y, predictions) - self.l2_reg * predictions

            # Train a decision tree on the residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Store the trained tree
            self.trees.append(tree)

            # Update predictions
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * tree_predictions

            # Update feature importances
            self.feature_importances_ += tree.feature_importances_

            # Check validation loss for early stopping
            if validation_data:
                X_val, y_val = validation_data
                val_predictions = self.predict_proba(X_val)
                val_loss = -np.mean(
                    y_val * np.log(val_predictions + 1e-9) + (1 - y_val) * np.log(1 - val_predictions + 1e-9)
                )
                if val_loss < best_loss:
                    best_loss = val_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Early stopping if no improvement is observed
                if no_improvement_count >= self.early_stopping_rounds:
                    print(f"Early stopping at iteration {iteration}.")
                    break

        # Normalize feature importances
        self.feature_importances_ /= len(self.trees)

    def predict_proba(self, X):
        """
        Predict probabilities for binary classification.
        
        Parameters:
        - X: Input features.
        
        Returns:
        - Probabilities for the positive class.
        """
        predictions = np.full(len(X), self.base_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return 1 / (1 + np.exp(-predictions))  # Apply sigmoid function

    def predict(self, X):
        """
        Make binary predictions.
        
        Parameters:
        - X: Input features.
        
        Returns:
        - Binary predictions (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)