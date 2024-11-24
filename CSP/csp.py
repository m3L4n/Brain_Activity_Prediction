# https://en.wikipedia.org/wiki/Common_spatial_pattern
# https://github.com/mne-tools/mne-python/blob/main/mne/decoding/csp.py

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    @staticmethod
    def _cov(X):
        """Compute covariance of a matrice (n_channel , ntime * n_epoch)

        Return X ndarray (n_channel , ntime * n_epoch)

        Parameter:
        X : ndarray (n_channel , ntime * n_epoch)
        """
        res = np.array(np.dot(X, X.T) / X.shape[1])
        return res

    @staticmethod
    def _compute_covariance_matrices(X):
        """
        We have this (n_epoch, n_channels, n_time)
        but we want this [ n_channels, n_epochs * n_times]
        for extract feature and compute cov

        return X the cov matrice of X ndarray (n_channel , ntime * n_epoch)

        Parameter:
        X : ndarray (n_channel , ntime , n_epoch)
        """
        idx_channel = 1
        idx_epoch = 0
        idx_time = 2
        n_epoch, n_channels, n_times = X.shape
        x_transposed = np.transpose(X, (idx_channel, idx_epoch, idx_time))
        X_reshape = x_transposed.reshape(n_channels, n_epoch * n_times)
        return CSP._cov(X_reshape)

    @staticmethod
    def _order_components(eigen_values):
        """
        sort eigen value in alternate mode
        sort eigen value in ascendent order and separate in twice part the vector
        at the pair index of eigenvalue insert the second part of the vector
        and at impair the first part of the vector

        return  ndaray of index of sorted eigen values

        Parameters:
        eigen_values : ndarray

        """
        i = np.argsort(eigen_values)
        ix = np.empty_like(i)
        ix[1::2] = i[: len(i) // 2]
        ix[0::2] = i[len(i) // 2 :][::-1]
        return ix

    def fit(self, X, y):
        """Separate class to fit the best distinction between R1 et R2
        and compute weight to transform other data to process a classifier on it

        Parameters:
        X : ndarray ( n_epoch, n_channels, n_times) the data
        y : ndarray (n_epoch) the tag of data
        """
        self._classes = np.unique(y)
        data_class_1, data_class_2 = self.define_class_data(X, y)
        R1 = self._compute_covariance_matrices(data_class_1)
        R2 = self._compute_covariance_matrices(data_class_2)
        eigen_values, eigen_vector = eigh(R1, R1 + R2)
        ix = self._order_components(eigen_values)

        eigen_vector = eigen_vector[:, ix]
        self.filters_ = eigen_vector.T[: self.n_components]

        return self

    def transform(self, X):
        """Transform our data X with the weight filter and return it

        Parameters:
        X : ndarray ( n_epoch, n_channels, n_times)
        the data that we want to trasnform
        """
        X = np.array(X)
        X_transformed = np.array([np.dot(self.filters_, x) for x in X])
        X = (X_transformed**2).mean(axis=2)
        X = np.log(X)

        return X

    def define_class_data(self, X, labels_array):
        """Separate our data in two distinct class (class 1 T1 et class 2 T2).

        return X1 (( n_epoch, n_channels, n_times)) and X2 ( n_epoch, n_channels, n_times)

        Parameters:
        X : ndarray ( n_epoch, n_channels, n_times) the data
        labels_array : ndarray (n_epoch) the tag of data
        """
        X1 = X[labels_array == self._classes[0]]  # data class 0 (T1)
        X2 = X[labels_array == self._classes[1]]  # data class 1 (T2)
        return X1, X2
