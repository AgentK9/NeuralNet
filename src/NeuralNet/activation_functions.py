from typing import List

import numpy as np


def step_func(inputs: List[float]) -> List[float]:
    return [0 if x <= 0 else 1 for x in inputs]


def sigmoid_func(inputs: List[float]) -> List[float]:
    return [1/(1 + np.exp(-x)) for x in inputs]


def rec_linear_func(inputs: List[float]) -> List[float]:
    return [np.maximum(0, x) for x in inputs]


def softmax_func(inputs: List[float]) -> List[float]:
    return np.exp(inputs) / np.sum(inputs)
