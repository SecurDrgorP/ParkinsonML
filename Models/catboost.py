from sklearn.tree import DecisionTreeRegressor
import numpy as np

class CatBoostScratch:
    def __init__(self, n_estimators=500, learning_rate=0.1, max_depth=3, l2_reg=0.1):
        """
        Initialize CatBoost parameters.
        
        Parameters:
        - n_estimators: Number of trees (boosting rounds).
        - learning_rate: Step size for updating predictions.
        - max_depth: Depth of each decision tree.
        - l2_reg: L2 regularization to reduce overfitting.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.trees = []  # Stores all decision trees
        self.base_prediction = None  # Base prediction (initial log-odds)
        self.feature_importances_ = None  # Feature importance tracker

    def _log_loss_gradient(self, y_true, y_pred):
        """
        Compute gradient of log-loss (cross-entropy) with respect to predictions.
        
        Parameters:
        - y_true: Actual labels.
        - y_pred: Current predictions (logits).
        
        Returns:
        - Gradient of log-loss.
        """
        sigmoid = 1 / (1 + np.exp(-y_pred))  # Apply sigmoid to logits
        return (y_true - sigmoid)  # Gradient calculation

    def fit(self, X, y):
        """
        Train the CatBoost model using boosting logic.
        
        Parameters:
        - X: Input features (scaled).
        - y: Target labels.
        """
        # Initialize base prediction as log-odds for binary classification
        self.base_prediction = np.log(np.mean(y) / (1 - np.mean(y)))
        predictions = np.full(len(y), self.base_prediction)  # Initialize all predictions

        # Initialize feature importance array
        self.feature_importances_ = np.zeros(X.shape[1])

        for _ in range(self.n_estimators):
            # Compute log-loss gradient (residuals)
            residuals = self._log_loss_gradient(y, predictions)

            # Train a decision tree on residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)

            # Store the fitted tree
            self.trees.append(tree)

            # Update predictions using the tree and apply L2 regularization
            tree_predictions = tree.predict(X)
            predictions += self.learning_rate * (tree_predictions - self.l2_reg * tree_predictions)

            # Accumulate feature importances
            self.feature_importances_ += tree.feature_importances_

        # Normalize feature importances for interpretability
        self.feature_importances_ /= self.n_estimators


    def predict(self, X):
        """
        Make predictions on new data.
        
        Parameters:
        - X: Input features.
        
        Returns:
        - Binary predictions (0 or 1).
        """
        # Start with base prediction (log-odds)
        predictions = np.full(len(X), self.base_prediction)

        # Aggregate predictions from all trees
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)

        # Convert logits to binary outputs using sigmoid and threshold at 0.5
        return (1 / (1 + np.exp(-predictions)) > 0.5).astype(int)

