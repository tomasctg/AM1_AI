import numpy as np
import numpy.linalg as LA
from BaseBayesianClassifier import BaseBayesianClassifier

class LDA(BaseBayesianClassifier):

  def _fit_params(self, X, y):
    
    n = len(y.flatten())
    n_j = np.bincount(y.flatten().astype(int))
    covs = [np.cov(X[:,y[0]==idx], bias=True)
                      for idx in range(len(self.log_a_priori))]
    
    self.shared_cov = sum(n_j[idx] * covs[idx] for idx in range(len(self.log_a_priori))) / n
          
    # Q6: por que se usa bias=True en vez del default bias=False?
    self.means = [X[:,y.flatten()==idx].mean(axis=1, keepdims=True)
                  for idx in range(len(self.log_a_priori))]
    # Q7: que hace axis=1? por que no axis=0?

  def _predict_log_conditional(self, x, class_idx):
    # predict the log(P(x|G=class_idx)), the log of the conditional probability of x given the class
    # this should depend on the model used
    unbiased_x =  x - 0.5*self.means[class_idx]
    return  self.means[class_idx].T @ self.shared_cov @ unbiased_x