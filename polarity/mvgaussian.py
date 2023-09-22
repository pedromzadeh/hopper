import numpy as np


class MVGaussian:
    def __init__(self, X):
        self.X = X.reshape(-1, 1, 2)

    def pdf(self, means, cov):
        rank = len(means)
        self._is_sym_pos_def(cov)
        # pre_factor = np.power(2 * np.pi, rank) * np.linalg.det(cov)
        # pre_factor = 1 / np.sqrt(pre_factor)
        pre_factor = 1

        inv_cov = np.linalg.inv(cov)
        term = (self.X - means) @ inv_cov @ (self.X - means).reshape(-1, 2, 1)
        return pre_factor * np.exp(-1 / 2 * term)

    def _is_sym_pos_def(self, cov):
        sym = np.all(cov == cov.T)
        pos_def = np.all(np.linalg.eigvals(cov) > 0)
        if not (sym and pos_def):
            raise ArithmeticError(
                "Covariance matrix must be symmetric positive definite."
            )
