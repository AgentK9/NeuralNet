from typing import List, Callable, Optional

import numpy as np
from numpy import dot


class Neuron:
    activation_func: Callable
    _bias: float
    _weights: Optional[np.array]

    def __init__(
        self,
        bias: float,
        act_func: Optional[Callable[[float, List[float]], float]] = None,
        weights: Optional[np.array] = None,
    ):
        self.activation_func = act_func or (lambda x, _: x)

        self.set_weights(weights)
        self._bias = bias

    def get_weights(self) -> np.array:
        return self._weights

    def set_weights(self, weights: np.array):
        self._weights = weights

    def forward(self, inputs: List[float]) -> float:
        if not self._weights:
            raise ValueError("Weights have not been initialized. Please initialize them.")
        # print(inputs, self._weights)
        # print(dot(inputs, self._weights))
        return dot(inputs, self._weights) + self._bias


if __name__ == '__main__':
    inputs = [[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]
    neurons = [
        Neuron(2, weights=[0.2, 0.8, -0.5, 1.0]),
        Neuron(3, weights=[0.5, -0.91, 0.26, -0.5]),
        Neuron(0.5, weights=[-0.26, -0.27, 0.17, 0.87])
    ]
    print([n.forward(inputs) for n in neurons])
