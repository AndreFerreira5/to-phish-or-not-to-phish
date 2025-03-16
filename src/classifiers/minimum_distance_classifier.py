import numpy as np


class MinimumDistanceClassifier:
    def __init__(self):
        self.classes = None
        self.centroids = None

    def fit(self, X, y):
        """
        Compute the centroid of each class.
        Args:
            X (numpy.ndarray): Training data (n_samples, n_features).
            y (numpy.ndarray): Class labels (n_samples,).
        """
        self.classes = np.unique(y)
        self.centroids = np.array([X[y == cls].mean(axis=0) for cls in self.classes])

    def predict(self, X):
        """
        Predict the class for each data point.
        Args:
            X (numpy.ndarray): Test data (n_samples, n_features).
        Returns:
            numpy.ndarray: Predicted class labels (n_samples,).
        """
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        return self.classes[np.argmin(distances, axis=0)]
