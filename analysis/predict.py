import numpy as np
import pandas as pd
from .analysis import get_bin_index


class PredictTrajectory:
    def __init__(self, X, V, F, Sigma):
        self.X = X
        self.V = V
        self.F = F
        self.Sigma = Sigma

    def integrate(self, x0, v0, T):
        x = [x0]
        v = [v0]
        delta_t = 0.00075 * 8 / 60  # hr

        for t in range(1, T):
            if np.isnan([x[t - 1], v[t - 1]]).any() or not (
                self.X.min() <= x[t - 1] <= self.X.max()
                and self.V.min() <= v[t - 1] <= self.V.max()
            ):
                break

            i, j = get_bin_index(x[t - 1], v[t - 1], self.X, self.V)[0]
            _x = x[t - 1] + v[t - 1] * delta_t
            _v = (
                v[t - 1]
                + self.F[i, j] * delta_t
                + self.Sigma[i, j] * np.random.randn() * np.sqrt(delta_t)
            )
            x.append(_x)
            v.append(_v)

        return x
