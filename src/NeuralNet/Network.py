import random
from typing import List, Optional, Callable

from Layer import Layer


class Network:
    _heights: List[int]
    _layers: List[Layer]

    def __init__(self, heights: List[int], act_func: Optional[Callable] = None):
        self._heights = heights
        self._layers = []
        # random neuron initialization
        for i, height in enumerate(self._heights):
            self._layers.append(
                Layer(
                    in_height=self._heights[max(0, i - 1)],
                    out_height=height,
                    act_func=act_func,
                )
            )

    def forward(self, inputs: List[float]) -> List[float]:
        if len(inputs) != self._heights[0]:
            raise ValueError(f"The length of Network.inputs should be the same as the first height "
                             f"({len(inputs)} != {self._heights[0]})")
        last_outputs: List[float] = inputs
        for i, layer in enumerate(self._layers):
            last_outputs = layer.forward(last_outputs)

        return last_outputs

    def batch_forward(self, inputs: List[List[float]]) -> List[List[float]]:
        return [self.forward(i) for i in inputs]


if __name__ == '__main__':
    randNet = Network([1, 3, 3, 2])
    x = randNet.forward([random.uniform(-10.0, 10.0)])
    print(x)
