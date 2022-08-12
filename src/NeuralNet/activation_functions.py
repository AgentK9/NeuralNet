from typing import List

import numpy as np


def step_func(X: float, _: List[float]) -> float:
    return 0 if X <= 0 else 1


def sigmoid_func(X: float, _: List[float]) -> float:
    return 1/(1 + np.exp(-X))


def rec_linear_func(X: float, _: List[float]) -> float:
    # print(X)
    return np.maximum(0, X)


def softmax_func(X: float, inputs: List[float]) -> List[float]:
    return np.exp(X - np.max(inputs)) / np.sum(np.exp(inputs))


if __name__ == '__main__':
    arr = [1, 2, 3]
    for x in arr:
        pass
        # print(softmax_func(x, arr))

