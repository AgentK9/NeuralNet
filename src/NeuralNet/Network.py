import random
from typing import List, Optional, Callable, Tuple

import numpy as np

from Layer import Layer


class Network:
    _heights: List[Tuple[int, Optional[Callable]]]
    _layers: List[Layer]

    def __init__(self, heights: List[Tuple[int, Optional[Callable]]]):
        self._heights = heights
        self._layers = []
        # random neuron initialization
        for i in range(1, len(self._heights)):
            height, act_func = self._heights[i]
            self._layers.append(
                Layer(
                    in_height=self._heights[max(0, i - 1)][0],
                    out_height=height,
                    act_func=act_func,
                )
            )

    def forward(self, inputs: List[float]) -> List[float]:
        if len(inputs) != self._heights[0][0]:
            raise ValueError(f"The length of Network.inputs should be the same as the first height "
                             f"({len(inputs)} != {self._heights[0][0]})")
        last_outputs: List[float] = inputs
        for layer in self._layers:
            # print(last_outputs)
            last_outputs = layer.forward(last_outputs)
        # print(last_outputs[0][0].shape)
        return last_outputs

    def batch_forward(self, inputs: List[List[float]]) -> List[List[float]]:
        return np.apply_along_axis(self.forward, axis=1, arr=inputs)


if __name__ == '__main__':
    randNet = Network([1, 3, 3, 2])
    x = randNet.forward([random.uniform(-10.0, 10.0)])
    # print(x)
