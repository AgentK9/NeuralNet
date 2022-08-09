import random
from typing import List, Callable, Optional

import numpy as np

from Neuron import Neuron


random_weight_range = (-1.0, 1.0)


class Layer:
    _neurons: List[Neuron]

    def __init__(self, in_height: int, out_height: int, act_func: Optional[Callable] = None):
        self._neurons = []
        for n in range(out_height):  # n_outputs
            # not sure if input neurons need to be specified....
            self._neurons.append(
                Neuron(
                    0.01 * np.random.randn(in_height),  # n_inputs
                    0,
                    act_func
                )
            )

    def forward(self, inputs: List[float]):
        outputs = [n.forward(inputs) for n in self._neurons]
        # print(inputs, outputs)
        results = []
        for n, out in zip(self._neurons, outputs):
            results.append(n.activation_func(out, outputs))
        return results
