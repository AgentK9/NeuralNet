from typing import Optional, Union

import numpy as np

from src.NeuralNet.activation.Activation import Activation


class Softmax(Activation):
    def forward(self, inputs: np.array) -> Optional[Union[float, np.array]]:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)
