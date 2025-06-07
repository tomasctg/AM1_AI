import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from scipy.linalg.lapack import dtrtri
from BaseBayesianClassifier import BaseBayesianClassifier


class QDA(BaseBayesianClassifier):

    def _fit_params(self, X, y):
        self.inv_covs = [LA.inv(np.cov(X[:, y[0] == idx], bias=True))
                         for idx in range(len(self.log_a_priori))]

        self.means = [X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
                      for idx in range(len(self.log_a_priori))]

    def _predict_log_conditional(self, x, class_idx):
        inv_cov = self.inv_covs[class_idx]
        unbiased_x = x - self.means[class_idx]
        return 0.5*np.log(LA.det(inv_cov)) - 0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        super()._fit_params(X, y)

        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(
            0, 2, 1) @ self.tensor_inv_cov @ unbiased_x

        return 0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class FasterQDA(TensorizedQDA):

    def predict(self, X):

        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=int)
        y_hat = self._predict_one(X)

        return y_hat.reshape(1, -1)

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori[:, np.newaxis] + self._predict_log_conditionals(x), axis=0)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(
            0, 2, 1) @ self.tensor_inv_cov @ unbiased_x
        diag_inner_prod = np.diagonal(inner_prod, axis1=1, axis2=2)

        return 0.5*np.log(LA.det(self.tensor_inv_cov))[:, np.newaxis] - 0.5 * diag_inner_prod


class EfficientQDA(TensorizedQDA):

    def predict(self, X):

        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=int)
        y_hat = self._predict_one(X)

        return y_hat.reshape(1, -1)

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori[:, np.newaxis] + self._predict_log_conditionals(x), axis=0)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means  # (k, p, n)
        inv_cov_x = self.tensor_inv_cov @ unbiased_x  # (k, p, n)
        diag_inner_prod = np.sum(inv_cov_x * unbiased_x, axis=1)  # (k, n)

        return 0.5 * np.log(np.linalg.det(self.tensor_inv_cov))[:, np.newaxis] - 0.5 * diag_inner_prod


class QDA_Chol1(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.L_invs = [
            LA.inv(
                cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True))
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
                      for idx in range(len(self.log_a_priori))]

    def _predict_log_conditional(self, x, class_idx):
        L_inv = self.L_invs[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = L_inv @ unbiased_x

        return np.log(L_inv.diagonal().prod()) - 0.5 * (y**2).sum()


class QDA_Chol2(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.Ls = [
            cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True)
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
                      for idx in range(len(self.log_a_priori))]

    def _predict_log_conditional(self, x, class_idx):
        L = self.Ls[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = solve_triangular(L, unbiased_x, lower=True)

        return -np.log(L.diagonal().prod()) - 0.5 * (y**2).sum()


class QDA_Chol3(BaseBayesianClassifier):
    def _fit_params(self, X, y):
        self.L_invs = [
            dtrtri(
                cholesky(np.cov(X[:, y.flatten() == idx], bias=True), lower=True), lower=1)[0]
            for idx in range(len(self.log_a_priori))
        ]

        self.means = [X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
                      for idx in range(len(self.log_a_priori))]

    def _predict_log_conditional(self, x, class_idx):
        L_inv = self.L_invs[class_idx]
        unbiased_x = x - self.means[class_idx]

        y = L_inv @ unbiased_x

        return np.log(L_inv.diagonal().prod()) - 0.5 * (y**2).sum()


class TensorizedChol(QDA_Chol3):

    def _fit_params(self, X, y):
        super()._fit_params(X, y)

        self.tensor_L_invs = np.stack(self.L_invs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        y = self.tensor_L_invs @ unbiased_x

        return np.sum(np.log(self.tensor_L_invs.diagonal(axis1=1, axis2=2)), axis=1) - 0.5 * np.sum(y ** 2, axis=(1, 2))

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class EfficientChol(TensorizedChol):

    def predict(self, X):

        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=int)
        y_hat = self._predict_one(X)

        return y_hat.reshape(1, -1)

    def _predict_one(self, x):
        return np.argmax(self.log_a_priori[:, np.newaxis] + self._predict_log_conditionals(x), axis=0)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        y = self.tensor_L_invs @ unbiased_x

        return np.sum(np.log(self.tensor_L_invs.diagonal(axis1=1, axis2=2)), axis=1)[:, np.newaxis] - 0.5 * np.sum(y ** 2, axis=1)
