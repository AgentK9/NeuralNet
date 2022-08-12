from typing import List

import numpy as np


class Loss:
    @staticmethod
    def _forward(output: List[float], y: List[float]):
        raise NotImplementedError

    @classmethod
    def calculate(cls, output: List[float], y: List[float]):
        sample_losses = cls._forward(output, y)
        actual_loss = np.mean(sample_losses)
        return actual_loss
