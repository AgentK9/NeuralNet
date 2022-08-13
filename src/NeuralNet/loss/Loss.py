from typing import List

import numpy as np


class Loss:
    @staticmethod
    def _forward(output: List[float], y: List[float]):
        raise NotImplementedError

    @classmethod
    def calculate(cls, y_pred: List[float], y_true: List[float]):
        sample_losses = cls._forward(y_pred, y_true)
        actual_loss = np.mean(sample_losses)
        return actual_loss
