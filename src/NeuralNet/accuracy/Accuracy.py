from typing import Union

import numpy as np


class Accuracy:
    @staticmethod
    def calculate(y_pred: np.array, y_true: np.array) -> Union[float, np.array]:
        predictions = np.argmax(y_pred, axis=1)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        return np.mean(predictions == y_true)
