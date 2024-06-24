# https://en.wikipedia.org/wiki/Common_spatial_pattern
import numpy as np
from scipy.linalg import eigh, inv
from sklearn.base import BaseEstimator, TransformerMixin


class CSP(TransformerMixin, BaseEstimator):

    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None
        
    def _cov_estimator(self, x_class, *, cov_kind, log_rank):
        _, n_channels, _ = x_class.shape

        x_class = x_class.transpose(1, 0, 2).reshape(n_channels, -1)
        cov = self._regularized_covariance(
            x_class,
            reg=None,
            method_params=self.cov_method_params,
            rank=self._rank,
            info=self._info,
            cov_kind=cov_kind,
            log_rank=log_rank,
            log_ch_type="data",
        )
        weight = x_class.shape[0]

        return cov, weight
        
    def _compute_covariance_matrices(self, X, y):
        # covs = []
        # sample_weights = []
        # for ci, this_class in enumerate(self._classes):
        #     cov, weight = self.cov_estimator(
        #         X[y == this_class],
        #         cov_kind=f"class={this_class}",
        #         log_rank=ci == 0,
        #     )

        #     if self.norm_trace:
        #         cov /= np.trace(cov)

        #     covs.append(cov)
        #     sample_weights.append(weight)

        # return np.stack(covs), np.array(sample_weights)
        pass
    def fit(self, x_array, epoch_array):
        self._classes = np.unique(epoch_array)
        # print("X array", x_array, "EPOCH\n", epoch_array)


        # covs, sample_weights = self._compute_covariance_matrices(X, y)
        # eigen_vectors, eigen_values = self._decompose_covs(covs, sample_weights)
        # ix = self._order_components(
        #     covs, sample_weights, eigen_vectors, eigen_values, self.component_order
        # )

        # eigen_vectors = eigen_vectors[:, ix]

        # self.filters_ = eigen_vectors.T











        # define two distinct data class
        data_class_1, data_class_2 = self.define_class_data(
            x_array, epoch_array)
        n_epochs = data_class_1.shape[0]
        n_channels = data_class_1.shape[1]
        n_times = data_class_1.shape[2]
        
        cov_matrices = np.zeros((n_channels, n_channels, n_epochs))
        for i in range(n_epochs): 
            X = data_class_1[i]  # Forme (n_channels, n_times)
            R = (X @ X.T) / n_times  # Calculer la matrice de covariance selon la formule donnée
            cov_matrices[:, :, i] = R
        cov_1 = np.mean(cov_matrices, axis=2)
        
        n_epochs = data_class_2.shape[0]
        n_channels = data_class_2.shape[1]
        n_times = data_class_2.shape[2]
        cov_matrices = np.zeros((n_channels, n_channels, n_epochs))
        for i in range(n_epochs): 
            X = data_class_2[i]  # Forme (n_channels, n_times)
            R = (X @ X.T) / n_times  # Calculer la matrice de covariance selon la formule donnée
            cov_matrices[:, :, i] = R
        cov_2 = np.mean(cov_matrices, axis=2)

        # calculate inverse of cov2
        R2_inv = inv(cov_2)
        
        # calculate * of inverse amd cov 1
        R = np.dot(R2_inv, cov_1)
        
        # calculate eighenvalues and vector
        eigvals, eigvecs  = eigh(R)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        m = self.n_components
        self.filters_ = np.hstack((eigvecs[:, :m], eigvecs[:, -m:]))
        print("SHape",self.filters_.shape)

        return self
    def transform(self, X):
        pick_filters = self.filters_[: self.n_components]
        X = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

    
        return X

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
