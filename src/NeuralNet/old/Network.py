from typing import List, Optional, Callable, Tuple, Type

import numpy as np

from Layer import Layer
from src.NeuralNet.loss.Loss import Loss


def set_random_weights(weights: np.array) -> np.array:
    return 0.01 * np.random.random_sample(weights.shape)


class Network:
    _heights: List[Tuple[int, Optional[Callable]]]
    _layers: List[Layer]
    _run_once: bool = True

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
                    # in_height=self._heights[i - 1][0],
                    out_height=height,
                    act_func=act_func,
                )
            )
        self.set_weights(weights=weights)

    def get_weights(self) -> np.array:
        return np.array([
            layer.get_weights()
            for layer in self._layers
        ])

    def set_weights(self, weights: Optional[np.array] = None):
        for i, layer in enumerate(self._layers):
            if weights is None:  # randomize the weights for this layer if the heights do not exist.
                layer.set_weights(0.01 * np.random.randn(self._heights[i+1][0]))
            else:
                layer.set_weights(weights[i])

    def optimize(self, n: int, n_plus_one_weights: Callable, x: np.array, y: np.array) -> float:
        lowest_loss = 99999
        best_weights = self.get_weights()
        best_accuracy = 0.0
        accuracy = 0.0
        for iteration in range(n):
            self.set_weights(n_plus_one_weights(self.get_weights()))
            self.batch_forward(x)
            loss = self.calculate_loss(y)
            predictions = np.argmax(self._output, axis=1)
            accuracy = np.mean(predictions == y)
            if loss < lowest_loss:
                print(f"New best weights at iter {iteration}\nLoss: {loss}\nAcc: {accuracy}")
                best_weights = self.get_weights()
                lowest_loss = loss
                best_accuracy = accuracy
            if iteration % (n/20) == 0:
                print(f"Iter {iteration}. Accuracy: {best_accuracy}")
        self.set_weights(best_weights)
        return best_accuracy

    def optimize_randomly(self, n: int, x: np.array, y: np.array) -> float:
        return self.optimize(n=n, n_plus_one_weights=set_random_weights, x=x, y=y)

    def forward(self, inputs: np.array) -> np.array:
        if len(inputs) != self._heights[0][0]:
            raise ValueError(f"The length of Network.inputs should be the same as the first height "
                             f"({len(inputs)} != {self._heights[0][0]})")
        last_outputs: np.array = inputs
        for layer in self._layers:
            if self._run_once:
                print(last_outputs)
            # print(last_outputs)
            last_outputs = layer.forward(last_outputs)
        if self._run_once:
            print(last_outputs)
            self._run_once = False
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
