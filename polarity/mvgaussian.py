import numpy as np


class MVGaussian:
    def __init__(self, means, cov):

        self.means = means
        self.cov = cov
        self.rank = len(means)
        self._is_sym_pos_def()
        pre_factor = np.power(2 * np.pi, self.rank) * np.linalg.det(cov)
        self.pre_factor = 1 / np.sqrt(pre_factor)

    def pdf(self, X):
        inv_cov = np.linalg.inv(self.cov)
        X = X.reshape(-1, 1, 2)
        term = (X - self.means) @ inv_cov @ (X - self.means).reshape(-1, 2, 1)
        return self.pre_factor * np.exp(-1 / 2 * term)

    def _is_sym_pos_def(self):
        sym = np.all(self.cov == self.cov.T)
        pos_def = np.all(np.linalg.eigvals(self.cov) > 0)
        if not (sym and pos_def):
            raise ArithmeticError(
                "Covariance matrix must be symmetric positive definite."
            )
