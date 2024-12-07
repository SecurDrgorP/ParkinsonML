import numpy as np


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