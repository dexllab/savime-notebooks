from abc import ABC, abstractmethod
import itertools
import json
import os
from typing import Callable, ClassVar, List, Union

from src.util import *

from tensorflow.keras.layers import BatchNormalization, ConvLSTM2D, Conv3D

from tqdm import tqdm as progress_bar


class MetricMatrixAndArray:
    def __init__(self, matrix, array):
        self.matrix = matrix
        self.array = array

    def __repr__(self):
        return f'{self.__class__.__name__}(matrix={self.matrix}, array={self.array})'


class ModelInterface(ABC):
    """
    An interface for keras models.

    """

    def __init__(self, name: str,
                 loss: tf.keras.losses.Loss = tf.keras.losses.mean_absolute_error,
                 optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(0.01),
                 metrics: Sequence[tf.metrics.Metric] = (tf.metrics.mean_absolute_error,),
                 epochs: int = 100, fit_batch_size: int = 32, fit_verbosity: int = 0,
                 evaluate_verbosity: int = 0,
                 early_stopping_min_delta: float = None,
                 early_stopping_patience: int = None):

        self.name = name
        self._model = None

        self.is_fit = False
        self.is_built = False
        self.is_compiled = False

        self._loss = loss
        self._optimizer = optimizer
        self._metrics = list(metrics)

        self._fit_epochs = epochs
        self._fit_batch_size = fit_batch_size
        self._fit_verbosity = fit_verbosity

        self._evaluate_verbosity = evaluate_verbosity

        self.metric_names = [metric.__name__ for metric in self._metrics]
        self._metrics_names_with_val_and_loss = self.metric_names + [f'val_{metric_name}'
                                                                     for metric_name in self.metric_names]
        self._metrics_names_with_val_and_loss.extend(['loss', 'val_loss'])
        self._fit_metric_watcher = MetricWatcher(self._metrics_names_with_val_and_loss)
        self._fit_callbacks = [self._fit_metric_watcher]

        self.iid = None

        early_stopping_config = {}
        if early_stopping_min_delta is not None:
            early_stopping_config['min_delta'] = early_stopping_min_delta
        if early_stopping_patience is not None:
            early_stopping_config['patience'] = early_stopping_patience

        if len(early_stopping_config) > 1:
            es_callback = tf.keras.callbacks.EarlyStopping(**early_stopping_config)
            self._fit_callbacks.append(es_callback)

    @abstractmethod
    def build(self):
        """
        This method must be overridden by the children classes. In this method, one has to implement the machine
        learning model. For instance, using the keras Sequence api.
        """
        pass

    def compile(self):
        """
        Compile a model, i.e., set the loss function, optimizer, and metrics.
        """
        if not self.is_built:
            self.build()
            self.is_built = True

        self._model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
        self.is_compiled = True

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            val_split: float = None, x_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Fit the model to a specific dataset.

        :param x_train: The input data used for  model training.
        :param y_train: The expected output data used for the model training.
        :param val_split: A value (between 0 and 1) denoting the percentage of data used for validation.
        :param x_val: The input data used for model validation.
        :param y_val: The output data used for model validation.
        """

        self._assert_validation(val_split, x_val, y_val)
        if not self.is_compiled:
            self.compile()
            self.is_compiled = True

        kwargs = dict(batch_size=self._fit_batch_size, epochs=self._fit_epochs, callbacks=self._fit_callbacks,
                      verbose=self._fit_verbosity)

        if x_val is not None:
            kwargs['validation_data'] = (x_val, y_val)
        elif val_split is not None:
            kwargs['validation_split'] = val_split

        self._model.fit(x_train, y_train, **kwargs)
        self.is_fit = True

    @staticmethod
    def _assert_validation(val_split, x_val, y_val):
        assert (val_split is None) ^ (x_val is None and y_val is None), 'You must either set a value between ' \
                                                                        'zero and one for the validation split or ' \
                                                                        'the inform the validation data.'
        assert not (x_val is None) ^ (y_val is None), 'The `x_val` must be informed if and only if th `y_val` has ' \
                                                      'been informed.'

    @assert_is_fit
    def get_fit_metrics(self) -> dict:
        return self._fit_metric_watcher.metrics

    @assert_is_fit
    def evaluate(self, x_test: np.ndarray, y_test: np.ndarray):
        """
        Evaluate the model. Wrapper for keras evaluate.

        :param x_test: The input data used for model testing.
        :param y_test: The expected data used for model testing.
        """

        return self._model.evaluate(x_test, y_test, verbose=self._evaluate_verbosity)

    @assert_is_fit
    def predict(self, x: np.ndarray):
        """
        Make predictions with the fit model. Wrapper for keras predict.

        :param x: The input data used for making predictions.
        """

        return self._model.predict(x)

    @assert_is_fit
    def save(self, output_dir: str, save_format='tf', version=1):
        """
        Save the fit model.

        :param output_dir: Output directory.
        :param save_format: The format to be saved. Choose either 'tf' or 'h5'.
        :param version: The model version.
        """

        export_path = os.path.join(output_dir, f'{self.name}/{version}')
        tf.keras.models.save_model(self._model, export_path, save_format=save_format)

    @assert_is_fit
    def save_metrics(self, output_dir: str):
        """
        Save the computed metrics (for fit stage).

        :param output_dir: Output directory.
        """

        metrics_fp = os.path.join(output_dir, f'{self.name}_metrics')
        with open(metrics_fp, 'w') as out:
            json.dump(self.get_fit_metrics, out)


class KerasArima(tf.keras.Model):
    """
    A Keras implementation of Arima. This implementation can be used to fit:
    - a 5th order tensor with dimensions (time_series_index, time_step, latitude, longitude, value).
    - a 3rd order tensor with dimensions (time_series_index, time_step, value)
    """

    def __init__(self, u=0., name=None):
        super().__init__(name=name)
        self.u = tf.constant(u, dtype=tf.double)
        self.phi = tf.Variable(tf.random.normal([1], dtype=tf.double))
        self.theta_1 = tf.Variable(tf.random.normal([1], dtype=tf.double))
        self.theta_2 = tf.Variable(tf.random.normal([1], dtype=tf.double))
        self.e_0 = tf.Variable(tf.random.normal([1], dtype=tf.double))

    def call(self, x, **kwargs):
        x = tf.convert_to_tensor(x)
        x = tf.cast(x, dtype=tf.double)

        num_time_steps = x.shape[1]

        x_0 = x[:, 0]
        x_1 = x[:, 1]
        y_0 = x_0 + self.u - self.theta_1 * self.e_0
        y_1 = x_1 + self.u + self.phi * (x_1 - x_0) - self.theta_1 * (x_1 - y_0) - self.theta_2 * self.e_0

        y = [y_0, y_1]

        for t in range(2, num_time_steps):
            x_t = x[:, t]
            x_t_min_1 = x[:, t - 1]
            y_t_min_1 = y[t - 1]
            y_t_min_2 = y[t - 2]
            y_t = self._y_t(x_t=x_t, x_t_min_1=x_t_min_1, y_t_min_1=y_t_min_1, y_t_min_2=y_t_min_2)
            y.append(y_t)

        return tf.stack(y, axis=1)

    @tf.function
    def _y_t(self, x_t, x_t_min_1, y_t_min_1, y_t_min_2):
        return x_t + self.u + self.phi * (x_t - x_t_min_1) - self.theta_1 * (x_t - y_t_min_1) - \
               self.theta_2 * (x_t_min_1 - y_t_min_2)


class Arima(ModelInterface):

    def __init__(self, name: str, u: float, **kwargs):
        super().__init__(name, **kwargs, metrics=[tf.keras.metrics.mean_absolute_error])
        self.u = u

    def build(self):
        self._model = KerasArima(u=self.u, name=self.name)


class TemperatureConvLSTM(ModelInterface):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self):
        self._model = tf.keras.Sequential(name=self.name)
        self._model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), input_shape=(None, 50, 50, 1),
                                   padding='same', return_sequences=True, use_bias=True))
        self._model.add(BatchNormalization())

        self._model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                                   use_bias=True))
        self._model.add(BatchNormalization())

        self._model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True,
                                   use_bias=True))
        self._model.add(BatchNormalization())

        self._model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), padding='same',
                               data_format='channels_last'))

        self._model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)


class ElasticNet(ModelInterface):

    def __init__(self, input_shape: tuple, **kwargs):
        super().__init__(**kwargs)

        self._kernel_constraint = tf.keras.constraints.unit_norm()
        self._kernel_regularizer = tf.keras.regularizers.l1_l2()
        self._input_shape = input_shape

    def build(self):
        self._model = tf.keras.Sequential()
        self._model.add(tf.keras.layers.Dense(1, input_shape=self._input_shape,
                                              kernel_constraint=self._kernel_constraint,
                                              kernel_regularizer=self._kernel_regularizer))


class ModelFactory:
    _model_name_formatter = '{prefix}_{id}'

    def __init__(self, model_class: ClassVar, x: np.ndarray, f_x: Union[Callable, np.ndarray], num_models: int,
                 test_size: float, val_size: float, model_name_prefix: str, f_kwargs: dict = None,
                 x_split_axis: int = None, y_split_axis: int = None,
                 model_kwargs: dict = None,
                 train_val_test_splitter: Callable = multidimensional_train_val_test):

        self._model_class = model_class
        self._model_kwargs = model_kwargs if model_kwargs is not None else {}
        self._x = x
        if callable(f_x):
            self._y = f_x(x) if f_kwargs is None else f_x(x, **f_kwargs)
        else:
            self._y = f_x

        self._num_partition_models = num_models
        self._num_partitions = num_models
        self._model_name_prefix = model_name_prefix
        self._test_size = test_size
        self._val_size = val_size

        if x_split_axis is None:
            self._x_splitter = ArraySplitter(self._x, self._num_partitions, axis=0)
            self._y_splitter = ArraySplitter(self._y, self._num_partitions, axis=0)
        else:
            assert x_split_axis is not None and y_split_axis is not None, 'You should provide both x_split_axis and ' \
                                                                          'y_split_axis'

            partitions = factor_number(num_models)
            self._x_splitter = ArraySplitter(self._x, partitions, axis=x_split_axis)
            self._y_splitter = ArraySplitter(self._y, partitions, axis=y_split_axis)

        self._train_val_test_splitter = train_val_test_splitter

        self._x_split = None
        self._y_split = None
        self._x_split_indices = None
        self._y_split_indices = None

        self._partition_test_data = []
        self._test_data = None

        self._partition_models: List[ModelInterface] = []
        self._model: ModelInterface = None
        self._all_models_iterator = None

    def _split(self):
        self._x_split, self._x_split_indices = self._x_splitter()
        self._y_split, self._y_split_indices = self._y_splitter()

    def build_models(self):
        """
        Instantiate and call the build model for each model.
        """
        def instantiate_and_compile(model_name):
            model_ = self._model_class(name=model_name, **self._model_kwargs)
            model_.compile()
            return model_

        for i in progress_bar(range(self._num_partition_models), desc='Building models'):
            name = self._model_name_formatter.format(prefix=self._model_name_prefix, id=i)
            model = instantiate_and_compile(name)
            model.iid = i
            self._partition_models.append(model)

        name = self._model_name_formatter.format(prefix=self._model_name_prefix, id=self._num_partition_models)
        self._model = instantiate_and_compile(name)
        self._model.iid = self._num_partition_models

        self._all_models_iterator = lambda: itertools.chain(self._partition_models, [self._model])

    def fit_models(self):
        """
        Iteratively fit each model in its partition.
        """

        self._split()
        for model_i, x_i, y_i in \
                progress_bar(zip(self._partition_models, self._x_split, self._y_split), desc='Training models by'
                                                                                             ' partition'):
            self._fit(model_i, x_i, y_i)

        for model in progress_bar([self._model], desc='Training a model in the whole domain'):
            self._fit(model, self._x, self._y, is_a_model_for_partition=False)

    def _fit(self, model: ModelInterface, x, y, is_a_model_for_partition=True):
        """
        Fit each model

        :model: The model to be fit.
        :x: The model input.
        :y: The model output.
        :is_a_model_for_partition: If the current model is being trained on a partition (True) or not (False).
        """
        x_train, y_train, x_val, y_val, x_test, y_test = self._train_val_test_splitter(x, y,
                                                                                       val_size=self._val_size,
                                                                                       test_size=self._test_size)
        model.fit(x, y, x_val=x_val, y_val=y_val)

        if is_a_model_for_partition:
            self._partition_test_data.append((x_test, y_test))
        else:
            self._test_data = (x_test, y_test)

    def get_metric_info(self, use_test_data: bool = False):

        def data_iterator():
            if use_test_data:
                return itertools.chain(self._partition_test_data, [self._test_data])
            return itertools.chain(zip(self._x_split, self._y_split), [(self._x, self._y)])

        metrics = ['loss'] + self._partition_models[0].metric_names

        total_num_models = self._num_partition_models + 1
        total_num_partitions = self._num_partitions + 1
        metric_matrix_dim = (total_num_models, total_num_partitions)

        info = {metric: np.zeros(metric_matrix_dim) for metric in metrics}

        for model in progress_bar(self._all_models_iterator(), desc=f'Building metric matrices'):
            self._fill_metric_info_for_model(model, metrics, data_iterator, info)

        return info

    @staticmethod
    def _fill_metric_info_for_model(model: ModelInterface, metrics, data_iterator, info):

        for partition_iid, (x_partition, y_partition) in enumerate(data_iterator()):
            evaluation = model.evaluate(x_partition, y_partition)
            for metric_iid, metric_name in enumerate(metrics):
                info[metric_name][model.iid][partition_iid] = evaluation[metric_iid]

    def _crate_model_config_list_str(self, output_dir):
        output_str = 'model_config_list {\n'
        config_strings = []
        for model in self._all_models_iterator():
            name = model.name
            export_path = os.path.join(output_dir, name + '/')
            config_str = '\tconfig {\n' + f'\t\tname: "{name}",\n\t\tbase_path: ' \
                                          f'"{export_path}"\n\t\tmodel_platform: "tensorflow"' + '}'
            config_strings.append(config_str)
        output_str += ',\n'.join(config_strings) + '\n}\n'
        return output_str

    def save_models(self, output_dir, save_format='tf'):
        for model in self._all_models_iterator():
            model.save(output_dir, save_format)

        model_config_list_str = self._crate_model_config_list_str(output_dir)
        model_config_fp = os.path.join(output_dir, 'models.config')
        with open(model_config_fp, 'w') as out_:
            out_.write(model_config_list_str)

    def save_data(self, output_dir, metrics: dict = None):

        x_name = 'x'
        y_name = 'y'

        splits = {model.name: {'x': x_split, 'y': y_split}
                  for model, x_split, y_split in
                  zip(self._partition_models, self._x_split_indices, self._y_split_indices)}
        model_names = [model.name for model in self._all_models_iterator()]
        model_name_to_iid = {model.name: model.iid for model in self._all_models_iterator()}

        data = {'iid': model_name_to_iid,
                'model_names': model_names, 'splits': splits, 'model': self._model.name,
                'x_file_name': x_name + '.npy', 'y_file_name': y_name + '.npy',
                'model_name_prefix': self._model_name_prefix, 'output_dir': output_dir}

        if metrics is not None:
            metrics_to_save = {key: value.tolist() for key, value in metrics.items()}
            data['metrics'] = metrics_to_save

        np.save(os.path.join(output_dir, x_name), self._x.astype(np.float64))
        np.save(os.path.join(output_dir, y_name), self._y.astype(np.float64))

        data_conf_output_fp = os.path.join(output_dir, 'data.json')

        with open(data_conf_output_fp, 'w') as out:
            json.dump(data, out, sort_keys=True, indent=4)
