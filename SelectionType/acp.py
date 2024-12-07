import numpy as np

class PCAFromScratch:
    def __init__(self, n_components):
        """
        Initialize PCA with the number of components to retain.
        :param n_components: Number of principal components to keep.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        """
        Fit the PCA model on the data.
        :param X: Input data, shape (n_samples, n_features).
        """
        # Step 1: Standardize the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Step 4: Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Step 5: Select top n_components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        """
        Transform the data using the fitted PCA model.
        :param X: Input data, shape (n_samples, n_features).
        :return: Transformed data, shape (n_samples, n_components).
        """
        if self.components is None:
            raise ValueError("The PCA model has not been fitted yet.")
        
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """
        Fit the PCA model and transform the data.
        :param X: Input data, shape (n_samples, n_features).
        :return: Transformed data, shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)