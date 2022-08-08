from typing import List, Callable, Optional

from numpy import dot, array


class Neuron:
    _activation_func: Callable
    _weights: array
    _bias: float

    def __init__(self, weights: List[float], bias: float, act_func: Optional[Callable] = None):
        self._activation_func = act_func or (lambda x: x)

        self._weights = array(weights)
        self._bias = bias

    def forward(self, inputs: List[float]) -> float:
        return self._activation_func((dot(self._weights, inputs) + self._bias))

    def norm_forward(self, inputs: List[float]) -> float:
        in_total = sum(inputs)
        norm_vals = [v/in_total for v in inputs]
        return self.forward(norm_vals)


if __name__ == '__main__':
    inputs = [1, 2, 3, 2.5]
    neurons = [
        Neuron([0.2, 0.8, -0.5, 1.0], 2),
        Neuron([0.5, -0.91, 0.26, -0.5], 3),
        Neuron([-0.26, -0.27, 0.17, 0.87], 0.5)
    ]
    print([n.forward(inputs) for n in neurons])
