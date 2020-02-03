import numpy as np
from numba import jit


def f_x(x: np.ndarray, mean=0, std=.1) -> np.ndarray:
    return np.abs(np.sum(x, axis=1)) + np.random.normal(mean, std, size=x.shape[0])


@jit(nopython=True)
def f2_x(x: np.ndarray, low: float, high: float, num_partitions: int, mean=0, std=2) -> np.ndarray:
    interval_points = np.linspace(low, high, num_partitions + 1)[1:]
    alphas = np.linspace(1, 3, num_partitions)
    alpha_matrix = np.ones(x.shape)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            alpha_ix = np.argmax(x[i].reshape(-1, 1) <= interval_points)
            alpha_matrix[i][j] = alphas[alpha_ix]

    y = np.sum(x * alpha_matrix, axis=1) + np.random.normal(mean, std)
    return y
