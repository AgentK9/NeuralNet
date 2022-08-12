import random
from typing import List, Optional, Callable, Tuple, Type

import numpy as np

from Layer import Layer
from src.NeuralNet.loss.Loss import Loss


class Network:
    _heights: List[Tuple[int, Optional[Callable]]]
    _layers: List[Layer]

    def __init__(
        self,
        heights: List[Tuple[int, Optional[Callable]]],
        loss: Type[Loss],
        weights: Optional[np.array] = None,
    ):
        self._heights = heights
        self._layers = []
        self._loss_function = loss
        self._output = None
        # random neuron initialization
        for i in range(1, len(self._heights)):
            height, act_func = self._heights[i]
            self._layers.append(
                Layer(
                    in_height=self._heights[i - 1][0],
                    out_height=height,
                    act_func=act_func,
                )
            )
        self.set_weights(weights)

    def set_weights(self, weights: Optional[np.array] = None):
        for i, layer in enumerate(self._layers):
            if not weights or not weights[i]:  # randomize the weights for this layer if the heights do not exist.
                layer.set_weights(0.01 * np.random.randn(self._heights[i - 1][0]))
            else:
                layer.set_weights(weights[i])

    def optimize(self):
        pass

    def optimize_randomly(self):
        pass

    def forward(self, inputs: np.array) -> np.array:
        if len(inputs) != self._heights[0][0]:
            raise ValueError(f"The length of Network.inputs should be the same as the first height "
                             f"({len(inputs)} != {self._heights[0][0]})")
        last_outputs: np.array = inputs
        for layer in self._layers:
            # print(last_outputs)
            last_outputs = layer.forward(last_outputs)
        # print(last_outputs[0][0].shape)
        self._output = last_outputs
        return self._output

    def calculate_loss(self, y):
        return self._loss_function.calculate(self._output, y)

    def calculate_accuracy(self, y: np.array) -> float:
        return np.mean(np.argmax(self._output, axis=1) == y)

    def batch_forward(self, inputs: np.array) -> np.array:
        self._output = np.apply_along_axis(self.forward, axis=1, arr=inputs)

        return self._output


if __name__ == '__main__':
    randNet = Network([1, 3, 3, 2])
    x = randNet.forward([random.uniform(-10.0, 10.0)])
    # print(x)
