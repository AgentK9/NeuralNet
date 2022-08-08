import random
from typing import List, Callable, Optional

from numpy import array

from Neuron import Neuron


random_weight_range = (-1.0, 1.0)


class Layer:
    _neurons: List[Neuron]

    def __init__(self, in_height: int, out_height: int, act_func: Optional[Callable] = None):
        self._neurons = []
        for n in range(out_height):
            # not sure if input neurons need to be specified....
            self._neurons.append(
                Neuron(
                    [random.uniform(*random_weight_range) for _ in range(in_height)],
                    0,
                    act_func
                )
            )

    def forward(self, inputs: List[float], start: Optional[bool] = None, normalized: bool = False):
        if start:
            return [self._neurons[i].forward(inputs) for i in range(len(self._neurons))]
        return [n.forward(inputs) if not normalized else n.norm_forward(inputs) for n in self._neurons]
