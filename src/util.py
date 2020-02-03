from functools import wraps
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tqdm

from typing import Sequence, Tuple


def train_test_val(x, y, test_size: float, val_size: float):
    assert 0 < test_size + val_size < 1

    test_plus_val_size = test_size + val_size
    x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, test_size=test_plus_val_size)
    x_test, x_val, y_test, y_val = train_test_split(x_test_val, y_test_val, test_size=val_size / test_plus_val_size)

    return x_train, y_train, x_test, y_test, x_val, y_val


class MetricWatcher(tf.keras.callbacks.Callback):

    def __init__(self, metrics_to_watch):
        super().__init__()
        self._metrics_to_watch = metrics_to_watch
        self.metrics = None

    def on_train_begin(self, logs=None):
        self.metrics = {metric: [] for metric in self._metrics_to_watch}

    def on_epoch_end(self, batch, logs=None):
        for metric_to_watch in self._metrics_to_watch:
            self.metrics[metric_to_watch].append(logs.get(metric_to_watch))


def is_fitted(f):
    @wraps(f)
    def wrapped(instance, *args, **kwargs):
        assert instance.is_fit, 'You should first fit the model.'
        return f(instance, *args, **kwargs)

    return wrapped


class RowArraySplitter:

    def __init__(self, array: np.ndarray, num_splits: int):
        self._array = array
        self._num_splits = num_splits
        self._is_split = False
        self._split_array = None
        self._split_indices = None

    def __call__(self) -> Tuple[Sequence[np.ndarray], Tuple[Tuple[int, int], ...]]:
        if not self._is_split:
            self._split_array = np.array_split(self._array, self._num_splits)
            self._split_indices = self._compute_split_indices(self._split_array)

        return self._split_array, self._split_indices

    @staticmethod
    def _compute_split_indices(split_array: Sequence[np.ndarray]) -> Tuple[Tuple[int, int], ...]:
        array_lengths = [len(array) for array in split_array]
        last_indices = list(np.cumsum(array_lengths))
        first_indices = list([0] + last_indices[:-1])
        return tuple(zip(first_indices, last_indices))


# def get_mean_squared_error_matrix(models, x, y, split_indices):
#     num_models = len(models)
#     mse_matrix = np.zeros((num_models, len(split_indices) + 1))
#     y_split = np.array_split(y, split_indices)
#
#     for i, model in tqdm.tqdm(enumerate(models), desc='Building MSE matrix'):
#         y_hat = model.predict(x)
#         y_hat_split = np.array_split(y_hat, split_indices)
#         for j, (y_j, y_hat_j) in enumerate(zip(y_split, y_hat_split)):
#             mse_matrix[i][j] = mean_squared_error(y_j, y_hat_j)
#
#     return mse_matrix


def get_mean_squared_error_matrix(models, x_split, y_split):
    num_models = len(models)
    mse_matrix = np.zeros((num_models, len(x_split)))
    mse_array = np.zeros((num_models, 1))

    x = np.concatenate(x_split, axis=0)
    y = np.concatenate(y_split, axis=0)

    for i, model in tqdm.tqdm(enumerate(models), desc='Building MSE matrix'):
        y_hat = model.predict(x)
        mse_array[i] = mean_squared_error(y, y_hat)
        for j, (x_j, y_j) in enumerate(zip(x_split, y_split)):
            y_hat_j = model.predict(x_j)
            mse_matrix[i][j] = mean_squared_error(y_j, y_hat_j)

    return mse_matrix, mse_array
