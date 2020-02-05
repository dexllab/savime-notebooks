import numpy as np
import requests


class PredictionConsumer:

    url_fmt = 'http://{host}:{port}/v{version}/models/{model_name}:predict'

    def __init__(self, host: str, port: int, model_name: str, version: int = 1):
        self.host = host
        self.port = port
        self.model_name = model_name
        self.version = version
        self.url = self.url_fmt.format(host=self.host, port=self.port, version=self.version, model_name=self.model_name)

    def predict(self, array: np.ndarray) -> np.ndarray:
        headers = {"content-type": "application/json"}
        r = requests.post(self.url, json=self._get_prediction_dict(array), headers=headers)
        if r.status_code == requests.codes.ok:
            return np.array(r.json()['predictions'])
        raise Exception(f'No prediction returned. {r.json()["error"]}')

    @staticmethod
    def _get_prediction_dict(array: np.ndarray) -> dict:
        return dict(instances=array.tolist(), signature_name='serving_default')