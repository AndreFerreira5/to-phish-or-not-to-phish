import numpy as np


class BaseMinimumDistanceClassifier:
    def __init__(self, distance="euclidean"):
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
        distances = self.compute_distances(X)
        return self.classes[np.argmin(distances, axis=1)]

    def compute_distances(self, X):
        raise NotImplementedError("Method not implemented, should be in subclasses.")


class EuclideanMinimumDistanceClassifier(BaseMinimumDistanceClassifier):
    def compute_distances(self, X):
        return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)


class MahalanobisMinimumDistanceClassifier(BaseMinimumDistanceClassifier):
    def __init__(self):
        super().__init__()
        self.inv_covariances = None
        self.covariances = None

    def fit(self, X, y):
        super().fit(X, y)
        self.inv_covariances = {}
        for cls in self.classes:
            class_data = X[y == cls]
            covariance_matrix = np.cov(class_data, rowvar=False)
            if np.linalg.det(covariance_matrix) == 0:
                covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-5
            self.inv_covariances[cls] = np.linalg.inv(covariance_matrix)


    '''
    def compute_distances(self, X):
        n_samples = X.shape[0]
        n_classes = self.centroids.shape[0]
        distances = np.zeros((n_samples, n_classes))

        for i in range(n_classes):
            diff = X - self.centroids[i]
            distances[:, i] = np.sum(diff @ self.inv_covariances[self.classes[i]] * diff, axis=1)

        return np.sqrt(distances)
    '''

    def compute_distances(self, X):
        n_samples, n_classes = X.shape[0], self.centroids.shape[0]
        d2 = np.empty((n_samples, n_classes))

        for i, cls in enumerate(self.classes):
            diff = X - self.centroids[i]
            d2[:, i] = np.sum(diff @ self.inv_covariances[cls] * diff, axis=1)

        # numerical guard: force negatives that are ~-1e-12 → 0
        d2 = np.clip(d2, a_min=0.0, a_max=None)
        return np.sqrt(d2)

