from typing import List

import numpy as np

from src.NeuralNet.loss.Loss import Loss


class CatCrossEntropy(Loss):
    @staticmethod
    def _forward(output: List[float], y: List[float]):
        return np.clip(-np.log(output[y]), 1e-7, 1-1e-7)


if __name__ == '__main__':
    print(CatCrossEntropy.calculate([0.7, 0.1, 0.2], [0.0]))
