import itertools
import json
import os

from typing import Callable, ClassVar

from src.util import *
import tqdm.keras


class ElasticNet:

    def __init__(self, input_shape: tuple, name: str,
                 kernel_constraint: tf.keras.constraints.Constraint = tf.keras.constraints.unit_norm(),
                 regularizer: tf.keras.regularizers.Regularizer = tf.keras.regularizers.l1_l2(),
                 optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(0.001),
                 loss: tf.keras.losses.Loss = tf.keras.losses.mean_squared_error,
                 metrics: Sequence[tf.metrics.Metric] = (tf.metrics.mean_squared_error,),
                 epochs: int = 10, fit_batch_size: int = 256, fit_verbosity: int = 0,
                 early_stopping_min_delta: float = 1e-8):

        self.name = name
        self.input_shape = input_shape
        self._model = None
        self.is_fit = False

        self._kernel_constraint = kernel_constraint
        self._kernel_regularizer = regularizer
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = list(metrics)

        self._fit_num_epochs = epochs
        self._fit_batch_size = fit_batch_size
        self._fit_verbosity = fit_verbosity

        self._metrics_names = list(itertools.chain(*[(metric.__name__, f'val_{metric.__name__}')
                                                     for metric in self._metrics]))
        self._fit_metric_watcher = MetricWatcher(self._metrics_names)
        self._fit_callbacks = [tf.keras.callbacks.EarlyStopping(min_delta=early_stopping_min_delta),
                               self._fit_metric_watcher]

    def build(self):
        self._model = tf.keras.Sequential()
        self._model.add(tf.keras.layers.Dense(1, input_shape=self.input_shape,
                                              kernel_constraint=self._kernel_constraint,
                                              kernel_regularizer=self._kernel_regularizer))
        self._model.compile(optimizer=self._optimizer,
                            loss=self._loss,
                            metrics=self._metrics)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray):
        if self._model is None:
            self.build()

        self._model.fit(x_train, y_train, validation_data=(x_val, y_val),
                        epochs=self._fit_num_epochs, batch_size=self._fit_batch_size, verbose=self._fit_verbosity,
                        callbacks=self._fit_callbacks)

        self.is_fit = True

    @is_fitted
    def predict(self, x: np.ndarray) -> np.ndarray:
        y_hat = self._model.predict(x)
        return y_hat

    @is_fitted
    def save(self, path: str, save_format: str = 'tf'):
        tf.keras.models.save_model(self._model, path, save_format=save_format)

    @is_fitted
    def get_fit_metrics(self) -> dict:
        return self._fit_metric_watcher.metrics

    @is_fitted
    def save(self, output_dir: str, save_format='tf'):
        export_path = os.path.join(output_dir, self.name + '/1')
        tf.keras.models.save_model(self._model, export_path, save_format=save_format)

    @is_fitted
    def save_metrics(self, output_dir: str):
        metrics_fp = os.path.join(output_dir, f'{self.name}_metrics')
        with open(metrics_fp, 'w') as out:
            json.dump(self.get_fit_metrics, out)


class ModelFactory:

    model_name_formatter = '{prefix}_{id}'

    def __init__(self, model_class: ClassVar, x: np.ndarray, f: Callable, num_models: int,
                 test_size: float, val_size: float, model_name_prefix: str, f_kwargs: dict = None):
        self._model_class = model_class
        self._x = x
        self._y = f(x) if f_kwargs is None else f(x, **f_kwargs)
        self._num_models = num_models
        self._model_name_prefix = model_name_prefix
        self._test_size = test_size
        self._val_size = val_size

        self._x_split = None
        self._y_split = None
        self._x_split_indices = None
        self._y_split_indices = None

        self._models = None

        assert len(x.shape) == 2
        self._num_observations, self._num_features = x.shape
        self._input_shape = (self._num_features, )

    def _split(self):
        x_splitter = RowArraySplitter(self._x, self._num_models)
        y_splitter = RowArraySplitter(self._y, self._num_models)

        self._x_split, self._x_split_indices = x_splitter()
        self._y_split, self._y_split_indices = y_splitter()

    def build_models(self):
        self._models = []
        for i in tqdm.tqdm(range(self._num_models), desc='Building models'):
            name = self.model_name_formatter.format(prefix=self._model_name_prefix, id=i)
            model = self._model_class(input_shape=self._input_shape, name=name)
            model.build()
            self._models.append(model)

    def fit_models(self):
        self._split()
        for model_i, x_i, y_i in tqdm.tqdm(zip(self._models, self._x_split, self._y_split), desc='Training models'):
            x_i_train, y_i_train, x_i_test, y_i_test, x_i_val, y_i_val = train_test_val(x_i, y_i,
                                                                                        test_size=self._test_size,
                                                                                        val_size=self._val_size)
            model_i.fit(x_i_train, y_i_train, x_i_val, y_i_val)

    def get_mean_squared_error_matrix(self):
        return get_mean_squared_error_matrix(models=self._models, x_split=self._x_split, y_split=self._y_split)

    def _crate_model_config_list_str(self, output_dir):
        output_str = 'model_config_list {\n'
        config_strings = []
        for model in self._models:
            name = model.name
            export_path = os.path.join(output_dir, name + '/')
            config_str = '\tconfig {\n' + f'\t\tname: "{name}",\n\t\tbase_path: ' \
                                          f'"{export_path}"\n\t\tmodel_platform: "tensorflow"' + '}'
            config_strings.append(config_str)
        output_str += ',\n'.join(config_strings) + '\n}\n'
        return output_str

    def save_models(self, output_dir, save_format='tf'):
        for model in self._models:
            model.save(output_dir, save_format)

        model_config_list_str = self._crate_model_config_list_str(output_dir)
        model_config_fp = os.path.join(output_dir, 'models.config')
        with open(model_config_fp, 'w') as out_:
            out_.write(model_config_list_str)

    def save_data(self, output_dir):
        def to_float_to_int(x): return [int(float(xi)) for xi in x]

        x_name = 'x'
        y_name = 'y'

        data_conf = {'splits': {model.name: {'x': to_float_to_int(x_split), 'y': to_float_to_int(y_split)}
                                for model, x_split, y_split in
                                zip(self._models, self._x_split_indices, self._y_split_indices)},
                     'x_file_name': x_name + '.npy',
                     'y_file_name': y_name + '.npy',
                     'model_name_prefix': self._model_name_prefix,
                     'output_dir': output_dir}

        np.save(os.path.join(output_dir, x_name), self._x.astype(np.float64))
        np.save(os.path.join(output_dir, y_name), self._y.astype(np.float64))

        data_conf_output_fp = os.path.join(output_dir, 'data.json')
        with open(data_conf_output_fp, 'w') as out:
            json.dump(data_conf, out, sort_keys=True, indent=4)
