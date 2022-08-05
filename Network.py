import random
from typing import List

from Neuron import Neuron


random_weight_range = (-5.0, 5.0)
random_bias_range = (-1.0, 1.0)


class Network:
    _heights: List[int]
    _neurons: List[List[Neuron]]

    def __init__(self, heights: List[int]):
        self._heights = heights
        self._neurons = []
        # random neuron initialization
        for i, height in enumerate(self._heights):
            layer: List[Neuron] = []
            for n in range(height):
                # not sure if input neurons need to be specified....
                if i == 0:
                    # idk seems like weighting the input neurons is weird but maybe not?
                    layer.append(Neuron([1.0], random.uniform(*random_bias_range)))
                    continue
                layer.append(
                    Neuron(
                        [random.uniform(*random_weight_range) for _ in range(self._heights[i - 1])],
                        random.uniform(*random_bias_range)
                    )
                )

            self._neurons.append(layer)

    def calculate(self, inputs: List[float]) -> List[float]:
        if len(inputs) != self._heights[0]:
            raise ValueError(f"The length of Netowrk.calculate should be the same as the first height "
                             f"({len(inputs)} != {self._heights[0]})")
        last_outputs: List[float] = inputs
        for layer in self._neurons:
            last_outputs = [n.calculate(last_outputs) for n in layer]

        return last_outputs


if __name__ == '__main__':
    randNet = Network([1, 3, 3, 2])
    x = randNet.calculate([random.uniform(-10.0, 10.0)])
    print(x)
