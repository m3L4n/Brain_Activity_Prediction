# https://en.wikipedia.org/wiki/Common_spatial_pattern
# https://github.com/mne-tools/mne-python/blob/main/mne/decoding/csp.py

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

# to do implement covariance matrice
#  to do implement ftt or wavelet transform
#  add read me to explain the projet
#  make plot function
# decompose fit method order component like csp class with ix
# remake define class to make it more clean
#  add commentary on each function


class CSP(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    @staticmethod
    def _compute_covariance_matrices(X):
        """
        We have this (n_epoch, n_channels, n_time)
        but we want this
        [ n_channels, n_epochs * n_times] for extract feature and compute cov
        """
        idx_channel = 1
        idx_epoch = 0
        idx_time = 2
        n_epoch, n_channels, n_times = X.shape
        x_transposed = np.transpose(X, (idx_channel, idx_epoch, idx_time))
        X_reshape = x_transposed.reshape(n_channels, n_epoch * n_times)
        return np.cov(X_reshape)

    def fit(self, X, y):
        """Separate class to fit the best distinction between R1 et R2
        and compute weight to transform other data to process a classifier on it"""

        self._classes = np.unique(y)
        data_class_1, data_class_2 = self.define_class_data(X, y)
        R1 = self._compute_covariance_matrices(data_class_1)
        R2 = self._compute_covariance_matrices(data_class_2)
        eigen_values, eigen_vector = eigh(R1, R1 + R2)
        i = np.argsort(eigen_values)
        ix = np.empty_like(i)
        ix[1::2] = i[: len(i) // 2]
        ix[0::2] = i[len(i) // 2 :][::-1]

        eigen_vector = eigen_vector[:, ix]
        self.filters_ = eigen_vector.T[: self.n_components]

        return self

    def transform(self, X):
        """ "Transform our data X with the weight filter and return it"""
        X = np.array(X)
        X_transformed = np.array([np.dot(self.filters_, x) for x in X])
        X = (X_transformed**2).mean(axis=2)
        X = np.log(X)

        return X

    def define_class_data(self, X, labels_array):
        """Separate our data in two distinct class (class 1 T! et class 2 T2)"""
        self.class_labels_ = np.unique(labels_array)
        X1 = X[labels_array == self.class_labels_[0]]  # data class 0 (T1)
        X2 = X[labels_array == self.class_labels_[1]]  # data class 1 (T2)
        return X1, X2
