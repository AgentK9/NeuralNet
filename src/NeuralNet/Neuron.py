from typing import List, Callable, Optional

from numpy import dot


class Neuron:
    activation_func: Callable
    _bias: float

    def __init__(
        self,
        weights: List[float],
        bias: float,
        act_func: Optional[Callable[[float, List[float]], float]] = None
     ):
        self.activation_func = act_func or (lambda x, _: x)
        # print(weights)
        self._weights = weights
        self._bias = bias

    def forward(self, inputs: List[float]) -> float:
        # print(inputs, self._weights)
        # print(dot(inputs, self._weights))
        return dot(inputs, self._weights) + self._bias



if __name__ == '__main__':
    inputs = [[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]
    neurons = [
        Neuron([0.2, 0.8, -0.5, 1.0], 2),
        Neuron([0.5, -0.91, 0.26, -0.5], 3),
        Neuron([-0.26, -0.27, 0.17, 0.87], 0.5)
    ]
    # print([n.forward(inputs) for n in neurons])
