from functools import wraps
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import tqdm

from typing import Sequence, Tuple, Union


def read_numpy_array_from_disk(file_path: str) -> np.ndarray:
    return np.load(file_path)


def export_numpy_array_to_c_array(array: np.ndarray, c_data_type, file_path: str):
    array.astype(c_data_type).tofile(file_path)


def factor_number(number):
    factors = [i for i in range(1, number+1) if number % i == 0]
    divisibles = [(factor_i, factor_j, abs(factor_i - factor_j))
                  for factor_i in factors for factor_j in factors if factor_i * factor_j == number]

    divisibles = sorted(divisibles, key=lambda x: x[-1])

    return divisibles[0][:-1]


def plot_heatmap(matrix, title, x_label, y_label, ax=None, heatmap_kwargs: dict = None):
    if ax is None:
        ax = plt.axes()
    if heatmap_kwargs is None:
        heatmap_kwargs = {}

    sns.heatmap(matrix, ax=ax, **heatmap_kwargs)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def multidimensional_train_val_test(x, y, val_size: float = .2, test_size: float = .1):
    assert 0 < test_size + val_size < 1

    test_plus_val_size = test_size + val_size
    x_train, x_test_val, y_train, y_test_val = train_test_split(x, y, test_size=test_plus_val_size)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=test_size / test_plus_val_size)

    return x_train, y_train, x_val, y_val, x_test, y_test


def temporal_train_val_test(x, val_size: float = .2, test_size: float = .1):
    assert 0 < val_size + test_size < 1

    num_time_series = x.shape[0]
    time_series_len = x.shape[1]
    permutation = np.random.permutation(num_time_series)
    x_permutation = x[permutation, :]

    val_ix = num_time_series - int(np.ceil(num_time_series * (val_size + test_size)))
    test_ix = num_time_series - int(np.ceil(num_time_series * test_size))

    x_train = x_permutation[:val_ix, :time_series_len - 1]
    y_train = x_permutation[:val_ix, 1: time_series_len]
    x_val = x_permutation[val_ix: test_ix, :time_series_len - 1]
    y_val = x_permutation[val_ix: test_ix, 1: time_series_len]
    x_test = x_permutation[test_ix:, :time_series_len - 1]
    y_test = x_permutation[test_ix:, 1: time_series_len]

    return x_train, y_train, x_val, y_val, x_test, y_test


def temporal_train_val_test2(x, y, val_size: float = .2, test_size: float = .1):
    assert 0 < val_size + test_size < 1

    num_groups = x.shape[0]
    permutation = np.random.permutation(num_groups)
    x_permutation = x[permutation, :]
    y_permutation = y[permutation, :]    

    val_ix = num_groups - int(np.ceil(num_groups * (val_size + test_size)))
    test_ix = num_groups - int(np.ceil(num_groups * test_size))
    
    val_ix = num_groups - int(np.ceil(num_groups * (val_size + test_size)))
    test_ix = num_groups - int(np.ceil(num_groups * test_size))

    x_train = x_permutation[:val_ix]
    y_train = y_permutation[:val_ix]
    x_val = x_permutation[val_ix: test_ix]
    y_val = y_permutation[val_ix: test_ix]
    x_test = x_permutation[test_ix:]
    y_test = y_permutation[test_ix:]

    return x_train, y_train, x_val, y_val, x_test, y_test


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


def assert_is_fit(f):
    @wraps(f)
    def wrapped(instance, *args, **kwargs):
        assert instance.is_fit, 'You should first fit the model.'
        return f(instance, *args, **kwargs)
    return wrapped


class ArraySplitter:

    def __init__(self, array: np.ndarray, num_splits: Union[int, Tuple[int, int]],
                 axis: Union[int, Tuple[int, int]]):
        self._array = array
        self._num_splits = num_splits
        self._is_split = False
        self._split_arrays = None
        self._split_indices = None
        self._axis = axis

        self._is_1d_split = isinstance(self._axis, int)

    def __call__(self):
        self._assert_input()

        if not self._is_split:
            self._split_arrays = self._compute_split()
            self._split_indices = self._compute_split_indices(self._split_arrays)

        return self._split_arrays, self._split_indices

    def _assert_input(self):
        def check_int():
            return isinstance(self._axis, int) and isinstance(self._num_splits, int)

        def check_tuple():
            return isinstance(self._axis, tuple) and isinstance(self._num_splits, tuple)

        assert check_int() ^ check_tuple(), 'Only 1d (int) and 2d (tuple[int, int]) splits are supported.'

    def _compute_split(self):

        if self._is_1d_split:
            return np.array_split(self._array, self._num_splits, axis=self._axis)

        horizontal_split = np.array_split(self._array, self._num_splits[1], axis=self._axis[1])
        concat_horizontal_split = np.concatenate(horizontal_split, axis=self._axis[0])
        vertical_and_horizontal_split = np.array_split(concat_horizontal_split, np.product(self._num_splits),
                                                       axis=self._axis[0])

        return vertical_and_horizontal_split

    def _compute_split_indices(self, split_arrays: Sequence[np.ndarray]) -> Sequence[Tuple[int, int]]:
        def compute_for_axis(axis):
            array_lengths = [int(array.shape[axis]) for array in split_arrays]
            last_indices = np.cumsum(array_lengths).tolist()
            first_indices = list([0] + last_indices[:-1])
            return first_indices, last_indices

        if self._is_1d_split:
            split_indices = list(zip(compute_for_axis(self._axis)))
        else:
            # Todo: Fix this computation
            axis_0_first_indices, axis_0_last_indices = compute_for_axis(self._axis[0])
            axis_1_first_indices, axis_1_last_indices = compute_for_axis(self._axis[1])
            split_indices = list(zip(axis_0_first_indices, axis_0_last_indices, axis_1_first_indices,
                                     axis_1_last_indices))

        return split_indices


def get_mean_squared_error_matrix(models, x_split, y_split):
    num_models = len(models)
    mse_matrix = np.zeros((num_models, len(x_split)))
    mse_array = np.zeros((num_models, 1))

    x = np.concatenate(x_split, axis=0)
    y = np.concatenate(y_split, axis=0)

    for i, model in tqdm.tqdm(enumerate(models), desc='Building MSE matrix'):
        print(model.evaluate(x, y))
        y_hat = model.predict(x)
        mse_array[i] = mean_squared_error(y, y_hat)
        for j, (x_j, y_j) in enumerate(zip(x_split, y_split)):
            y_hat_j = model.predict(x_j)
            mse_matrix[i][j] = mean_squared_error(y_j, y_hat_j)

    return mse_matrix, mse_array
