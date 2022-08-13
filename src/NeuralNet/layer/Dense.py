from typing import Optional, Union

import numpy as np


class Dense:
    _n_inputs: int
    _n_neurons: int

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value: np.array):
        self._weights = value

    def randomize_weights(self):
        self._weights = 0.05 * np.random.randn(self._n_inputs, self._n_neurons)

    def randomly_adjust_weights(self):
        self._weights += 0.05 * np.random.randn(self._n_inputs, self._n_neurons)

    @property
    def biases(self):
        return self._biases

    @biases.setter
    def biases(self, value: np.array):
        self._biases = value

    def randomize_biases(self):
        self._biases = 0.05 * np.random.randn(1, self._n_neurons)

    def randomly_adjust_biases(self):
        self._biases += 0.05 * np.random.randn(1, self._n_neurons)

    _output: Optional[Union[float, np.array]]

    def __init__(self, n_inputs: int, n_neurons: int):
        self._n_inputs = n_inputs
        self._n_neurons = n_neurons
        self._weights = []
        self.randomize_weights()
        self._biases = np.zeros((1, n_neurons))
        self._output = None

    def forward(self, inputs: np.array) -> Union[float, np.array]:
        self._output = np.dot(inputs, self._weights) + self._biases
        return self._output

    def get_output(self) -> Optional[Union[float, np.array]]:
        return self._output
