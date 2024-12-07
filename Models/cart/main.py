import numpy as np
import pandas as pd
from collections import Counter
from cart.treeVisulizer import TreeVisualizer

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.samples = 0
        self.distribution = {}
        self.confidence = 0.0

class CARTClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_names = None
    
    def fit(self, X, y):
        """Train the decision tree with support for both numpy arrays and pandas DataFrames"""
        # Convert pandas Series to numpy array if necessary
        if isinstance(y, pd.Series):
            y = y.values
            
        # Handle pandas DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            # If X is numpy array, create default feature names
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        self.n_classes = len(set(y))
        self.root = self._grow_tree(X, y)
        return self
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # Create node and add metadata
        node = Node()
        node.samples = len(y)
        node.distribution = dict(Counter(y))
        if len(y) > 0:
            majority_class = max(node.distribution.items(), key=lambda x: x[1])[0]
            node.confidence = node.distribution[majority_class] / len(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(set(y)) == 1:
            node.value = self._most_common_label(y)
            return node
        
        # Find the best split
        best_feature, best_threshold = self._best_split(X, y)
        if best_feature is None:
            node.value = self._most_common_label(y)
            return node
        
        # Create child splits
        left_idxs = X[:, best_feature] < best_threshold
        right_idxs = ~left_idxs
        
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        node.right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        
        return node
    
    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_idxs = X[:, feature] < threshold
                right_idxs = ~left_idxs
                
                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_idxs], y[right_idxs])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    
        return best_feature, best_threshold
    
    def _information_gain(self, parent, left_child, right_child):
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        
        gain = self._gini_impurity(parent) - (
            weight_left * self._gini_impurity(left_child) +
            weight_right * self._gini_impurity(right_child)
        )
        return gain
    
    def _gini_impurity(self, y):
        counter = Counter(y)
        impurity = 1
        for count in counter.values():
            prob = count / len(y)
            impurity -= prob ** 2
        return impurity
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """Predict class for X"""
        # Handle pandas DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def convert_to_viz_format(self, node=None):
        """Convert the tree to the format expected by TreeVisualizer"""
        if node is None:
            node = self.root
            
        if node.value is not None:  # Leaf node
            return {
                'type': 'leaf',
                'prediction': node.value,
                'samples': node.samples,
                'distribution': node.distribution,
                'probability': node.confidence
            }
        else:  # Split node
            return {
                'type': 'split',
                'feature': self.feature_names[node.feature],
                'samples': node.samples,
                'distribution': node.distribution,
                'confidence': node.confidence,
                'split_type': 'continuous',
                'split_point': node.threshold,
                'children': {
                    f'≤ {node.threshold:.2f}': self.convert_to_viz_format(node.left),
                    f'> {node.threshold:.2f}': self.convert_to_viz_format(node.right)
                }
            }
    
    def visualize(self, figsize=(20, 12)):
        """Create and display a visualization of the tree"""
        viz_tree = self.convert_to_viz_format()
        visualizer = TreeVisualizer(viz_tree, figsize=figsize)
        return visualizer.visualize()


TreeVisualizer = TreeVisualizer  # Using the TreeVisualizer class