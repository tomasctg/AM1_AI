import numpy as np
import numpy.linalg as LA
from scipy.linalg import cholesky, solve_triangular
from BaseBayesianClassifier import BaseBayesianClassifier


class QDA(BaseBayesianClassifier):

    def _fit_params(self, X, y):
        # estimate each covariance matrix
        self.inv_covs = [LA.inv(np.cov(X[:, y[0] == idx], bias=True))
                         for idx in range(len(self.log_a_priori))]
        # Q5: por que hace falta el flatten y no se puede directamente X[:,y==idx]?
        # Dado que estamos sacando un slice de la matrix correspondiente a las observacion cuyo output
        # corresponde al correspondiente label. Entonces dado que la forma de la matrix X es (features,obeservacion)
        # con un valor en y asociado, buscamos filtrar esta matrix para obetner solo las observacion de cada label, por lo cual
        # X[:,y.flatten()==idx]] se lee como todas las filas (features) y solo las columnas (observaciones) que correspondan a un label
        # idx. Dado que y es una matrix (np.array) con dimencion (1,n) con columna (label), es necesario convertirlo en una vector fila
        # con el valor booleano correspondiente para
        # el filtrado o no de cada columna. Esto asi porque es es como espera slicing X[filas,columnas]. con filas y columnas como el
        # rango de valores continuos de una lista.
        """
    La necesidad de usar .flatten() se debe a cómo funciona el indexado booleano en NumPy.
    En este caso, la matriz X tiene forma (features, observaciones), es decir, cada columna representa una observación y cada fila representa una característica. 
    Por otro lado, el vector de etiquetas y tiene forma (1, 178) o (178, 1), es decir, es una matriz 2D en lugar de un vector plano. Si usamos directamente y == idx, el resultado es también una matriz 2D, con forma (1, 178) o (178, 1).
    Sin embargo, cuando usamos indexación como X[:, y == idx], NumPy espera que el índice booleano sea un vector 1D del mismo tamaño que el número de columnas de X, en este caso, 178. Si le pasamos una matriz 2D, lanza un error o da resultados inesperados.
    """
        # Q6: por que se usa bias=True en vez del default bias=False?
        self.means = [X[:, y.flatten() == idx].mean(axis=1, keepdims=True)
                      for idx in range(len(self.log_a_priori))]
        # Q7: que hace axis=1? por que no axis=0?

    def _predict_log_conditional(self, x, class_idx):
        # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
        # this should depend on the model used
        inv_cov = self.inv_covs[class_idx]
        unbiased_x = x - self.means[class_idx]
        return 0.5*np.log(LA.det(inv_cov)) - 0.5 * unbiased_x.T @ inv_cov @ unbiased_x


class TensorizedQDA(QDA):

    def _fit_params(self, X, y):
        # ask plain QDA to fit params
        super()._fit_params(X, y)

        # stack onto new dimension
        self.tensor_inv_cov = np.stack(self.inv_covs)
        self.tensor_means = np.stack(self.means)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(
            0, 2, 1) @ self.tensor_inv_cov @ unbiased_x

        return 0.5*np.log(LA.det(self.tensor_inv_cov)) - 0.5 * inner_prod.flatten()

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori + self._predict_log_conditionals(x))


class FasterQDA(TensorizedQDA):

    def predict(self, X):

        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=int)
        y_hat = self._predict_one(X)

        return y_hat.reshape(1, -1)

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori[:, np.newaxis] + self._predict_log_conditionals(x), axis=0)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means
        inner_prod = unbiased_x.transpose(
            0, 2, 1) @ self.tensor_inv_cov @ unbiased_x
        # Aca aparece (k,n,n)
        diag_inner_prod = np.diagonal(inner_prod, axis1=1, axis2=2)

        return 0.5*np.log(LA.det(self.tensor_inv_cov))[:, np.newaxis] - 0.5 * diag_inner_prod


class EfficientQDA(TensorizedQDA):

    def predict(self, X):

        m_obs = X.shape[1]
        y_hat = np.empty(m_obs, dtype=int)
        y_hat = self._predict_one(X)

        return y_hat.reshape(1, -1)

    def _predict_one(self, x):
        # return the class that has maximum a posteriori probability
        return np.argmax(self.log_a_priori[:, np.newaxis] + self._predict_log_conditionals(x), axis=0)

    def _predict_log_conditionals(self, x):
        unbiased_x = x - self.tensor_means  # (k, p, n)
        inv_cov_x = self.tensor_inv_cov @ unbiased_x  # (k, p, n)
        diag_inner_prod = np.sum(inv_cov_x * unbiased_x, axis=1)  # (k, n)
        
        return 0.5 * np.log(np.linalg.det(self.tensor_inv_cov))[:, np.newaxis] - 0.5 * diag_inner_prod


class QDA_Chol1(BaseBayesianClassifier):
  def _fit_params(self, X, y):
    self.L_invs = [
        LA.inv(cholesky(np.cov(X[:,y.flatten()==idx], bias=True), lower=True))
        for idx in range(len(self.log_a_priori))
    ]

    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]

  def _predict_log_conditional(self, x, class_idx):
    L_inv = self.L_invs[class_idx]
    unbiased_x =  x - self.means[class_idx]

    y = L_inv @ unbiased_x

    return np.log(L_inv.diagonal().prod()) -0.5 * (y**2).sum()