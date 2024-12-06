import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, SelectKBest, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.preprocessing import StandardScaler
from EFSA.anova import ANOVASelector
import statsmodels.api as sm

class FeatureSelector:
    def __init__(self, X, y):
        """
        Initialize the Feature Selector with input features and target variable
        
        Parameters:
        -----------
        X : pandas DataFrame or numpy array
            Input features
        y : pandas Series or numpy array
            Target variable
        """
        # Ensure input is numpy array
        self.X = X.values if hasattr(X, 'values') else X
        self.y = y.values if hasattr(y, 'values') else y
        
        # Store feature names if available
        self.feature_names = (X.columns.tolist() if hasattr(X, 'columns') 
                               else [f'feature_{i}' for i in range(X.shape[1])])
        
        # Standardize features for some methods
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def backward_elimination(self, significance_level=0.05, verbose=True):
        """
        Perform backward elimination to select statistically significant features.

        Parameters:
        - significance_level (float): Threshold for p-value to determine feature significance.
        - verbose (bool): If True, prints progress and removed features.

        Returns:
        - selected_features (list): List of statistically significant features.
        - final_model (statsmodels.regression.linear_model.RegressionResults): Final fitted OLS model.
        """
        # Convert to DataFrame for statsmodels
        X_df = pd.DataFrame(self.X, columns=self.feature_names)
        
        # Add a constant (intercept) to the features
        X_with_const = sm.add_constant(X_df)
        
        # Get the initial list of features
        features = X_with_const.columns.tolist()
        
        while len(features) > 0:
            # Fit OLS model with current features
            X_temp = X_with_const[features]
            model = sm.OLS(self.y, X_temp).fit()
            
            # Get p-values for all features
            p_values = model.pvalues
            max_p_value = p_values.max()  # Find the highest p-value
            excluded_feature = p_values.idxmax()  # Feature with the highest p-value

            if max_p_value > significance_level:
                # Remove the least significant feature
                features.remove(excluded_feature)
                if verbose:
                    print(f"Removing '{excluded_feature}' with p-value {max_p_value:.4f}")
            else:
                # Stop when all features are significant
                break

        # Fit the final model with selected features
        selected_features = features
        final_model = sm.OLS(self.y, X_with_const[selected_features]).fit()

        if verbose:
            print("\nBackward Elimination Complete.")
            print(f"Selected Features: {selected_features[1:]}")  # Exclude the constant from the output
        
        return selected_features[1:], final_model  # Exclude 'const' in the returned list

    def embedded_method(self, k_features=50, alpha=1.0):
        """
        Lasso (L1) Embedded Feature Selection
        
        Parameters:
        -----------
        k_features : int, optional (default=50)
            Number of top features to select
        alpha : float, optional (default=1.0)
            Regularization strength for Lasso
        
        Returns:
        --------
        selected_features : list
            Names of selected features
        """
        # Create Lasso model
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(self.X_scaled, self.y)
        
        # Get feature importances (absolute coefficients)
        feature_importances = np.abs(lasso.coef_)
        
        # Get indices of top k features
        top_feature_indices = np.argsort(feature_importances)[-k_features:]
        
        # Get selected feature names
        selected_features = [self.feature_names[i] for i in top_feature_indices]
        
        return selected_features, feature_importances[top_feature_indices]

    def wrapper_method(self, k_features=50, estimator=None):
        """
        Recursive Feature Elimination (RFE) Wrapper Method
        
        Parameters:
        -----------
        k_features : int, optional (default=50)
            Number of top features to select
        estimator : sklearn estimator, optional (default=LogisticRegression)
            Base estimator for recursive feature elimination
        
        Returns:
        --------
        selected_features : list
            Names of selected features
        """
        # Use LogisticRegression as default estimator if not provided
        if estimator is None:
            estimator = LogisticRegression(max_iter=1000)
        
        # Create RFE selector
        rfe_selector = RFE(estimator=estimator, n_features_to_select=k_features)
        rfe_selector = rfe_selector.fit(self.X_scaled, self.y)
        
        # Get selected feature indices
        selected_indices = rfe_selector.get_support(indices=True)
        
        # Get selected feature names
        selected_features = [self.feature_names[i] for i in selected_indices]
        
        return selected_features

    def anova_test(self, k_features=15):
        """
        ANOVA F-Test for Feature Selection
        
        Parameters:
        -----------
        k_features : int, optional (default=15)
            Number of top features to select
        
        Returns:
        --------
        selected_features : list
            Names of selected features
        """
        selector = ANOVASelector(self.feature_names)
        selected_features, f_scores = selector.select_k_best(self.X, self.y, k_features)
        
        return selected_features, f_scores

    def comprehensive_feature_selection(self, k_features=50):
        """
        Combine multiple feature selection methods
        
        Parameters:
        -----------
        k_features : int, optional (default=50)
            Number of top features to select
        
        Returns:
        --------
        final_selected_features : list
            Comprehensive list of selected features
        """
        # Apply different feature selection methods
        anova_features, _ = self.anova_test(k_features//3)
        anova_features = set(anova_features)
        embedded_features, _ = self.embedded_method(k_features//3)
        embedded_features = set(embedded_features)
        wrapper_features = set(self.wrapper_method(k_features//3))
        
        # Combine and get unique features
        combined_features = list(anova_features.union(embedded_features, wrapper_features))
        
        return combined_features[:k_features]