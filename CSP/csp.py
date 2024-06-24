# https://en.wikipedia.org/wiki/Common_spatial_pattern
#https://github.com/mne-tools/mne-python/blob/main/mne/decoding/csp.py#L546

import numpy as np
from scipy.linalg import eigh, inv
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
        
    def _concat_epoch(self, X):
        _, n_channels, _ = X.shape
        x_class = X.transpose(1, 0, 2).reshape(n_channels, -1)
        return x_class
        
        
    def _compute_covariance_matrices(self, X):
        """
        """
        x_concat = self._concat_epoch(X)
        return np.cov(x_concat)
        
    def fit(self, x_array, epoch_array):
        self._classes = np.unique(epoch_array)

        # define two distinct data class
        data_class_1, data_class_2 = self.define_class_data(
            x_array, epoch_array)
        cov = []
        cov_1 = self._compute_covariance_matrices(data_class_1)
        cov_2 = self._compute_covariance_matrices(data_class_2)
        cov.append(cov_1)
        cov.append(cov_2)
        covs = np.stack(cov)
        # # calculate inverse of cov2
        R2_inv = inv(covs[1])
        product = np.dot(R2_inv,covs[0])
        eigenvalues, eigenvectors = eigh(product)
        ix = np.argsort(np.abs(eigenvalues - 0.5))[::-1]
        eigen_vectors = eigenvectors[:, ix]

        self.filters_ = eigen_vectors.T
        X = (x_array**2).mean(axis=2)
        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
        # To standardize features
        return self
    
    
    def transform(self, X):
        pick_filters = self.filters_[: self.n_components]
        X_class = np.asarray([np.dot(pick_filters, epoch) for epoch in X])
        X_class = (X_class**2).mean(axis=2)
        X_class = np.log(X_class)
        return X_class

    def define_class_data(self, X, labels_array):
        """ define class A (t1) class B (t2)
        """
        self.class_labels_ = np.unique(labels_array)
        X1 = X[labels_array == self.class_labels_[0]]  # data class 0 (T1)
        X2 = X[labels_array == self.class_labels_[1]]  # data class 1 (T2)
        return X1, X2

    def calculate_covariance_matrix(self, X):
        cov_matrix = np.cov(X)
        return cov_matrix
