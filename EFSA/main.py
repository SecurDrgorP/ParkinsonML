from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression
import numpy as np

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
    
    def wrapper_method(self, estimator=LogisticRegression(), k_features=5):
        """
        Recursive Feature Elimination (RFE) - Wrapper Method
        
        Parameters:
        -----------
        estimator : sklearn estimator, optional (default=LogisticRegression())
        k_features : int, optional (default=5)
            Number of features to select
        
        Returns:
        --------
        selected_features : list
            Names of selected features
        """
        pass
    
    
    def embedded_method(self, k_features=5):
        pass
    
    
    
    
    def chi_square_test(self, k_features=5):
        """
        Chi-Square Test for Feature Selection
        
        Parameters:
        -----------
        k_features : int, optional (default=5)
            Number of top features to select
        
        Returns:
        --------
        selected_features : list
            Names of selected features
        """
        # Ensure non-negative values for chi-square test
        X_positive = self.X - self.X.min()
        
        # Perform Chi-Square Test
        selector = SelectKBest(score_func=chi2, k=k_features)
        selector.fit(X_positive, self.y)
        
        # Get selected feature names
        selected_features = [self.feature_names[i] for i in selector.get_support(indices=True)]
        
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
    
    


class ANOVASelector:
    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def calculate_f_statistic(self, X, y):
        """
        Calculate F-statistic for each feature
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        
        Returns:
        f_scores: array of F-statistics for each feature
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        f_scores = np.zeros(n_features) # Initialize array to store F-scores
        
        for feature_idx in range(n_features):
            
            # Values of the feature 
            feature_values = X[:, feature_idx] # X[:, feature_idx] is the feature column
            
            # Calculate overall mean
            grand_mean = np.mean(feature_values)
            
            # Calculate between-group sum of squares (SSB) SSB = Σni(yi - ȳ)2
            # where yi is the mean of the class and ȳ is the grand mean
            
            
            ssb = 0
            for class_label in classes:
                class_mask = (y == class_label)
                class_values = feature_values[class_mask]
                class_mean = np.mean(class_values)
                class_size = len(class_values)
                
                ssb += class_size * (class_mean - grand_mean) ** 2
                
            # Calculate within-group sum of squares (SSW)  SSW = ΣΣ(yij - yi)2
            # where yij is the value of the j-th observation in the i-th class and yi is the mean of the i-th class
        
            ssw = 0
            for class_label in classes:
                class_mask = (y == class_label)
                class_values = feature_values[class_mask]
                class_mean = np.mean(class_values)
                
                ssw += np.sum((class_values - class_mean) ** 2)
            
            # Calculate degrees of freedom
            df_between = n_classes - 1
            df_within = n_samples - n_classes
            
            # Calculate mean squares
            msb = ssb / df_between
            msw = ssw / df_within
            
            # Calculate F-statistic
            f_scores[feature_idx] = msb / msw if msw != 0 else 0
            
        return f_scores
    
    def select_k_best(self, X, y, k):
        """
        Select k best features based on F-statistics
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        k: number of features to select
        
        Returns:
        selected_features: list of k best feature names
        """
        # Convert inputs to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Calculate F-statistics for all features
        f_scores = self.calculate_f_statistic(X, y)
        
        # Get indices of k features with highest F-scores
        
        top_k_indices = np.argsort(f_scores)[-k:] # argosrt returns the indices of the sorted array
        
        # Get feature names for selected features
        selected_features = [self.feature_names[i] for i in top_k_indices]
        
        return selected_features, f_scores

    def get_feature_scores(self, X, y):
        """
        Get F-scores for all features in the dataset
        using ANOVA F-test
    
        Parameters:
        X: array-like of shape (n_samples, n_features)
        y: array-like of shape (n_samples,)
        
        Returns:
        feature_scores: dictionary mapping feature names to F-scores
        """
        f_scores = self.calculate_f_statistic(X, y)
        return dict(zip(self.feature_names, f_scores))