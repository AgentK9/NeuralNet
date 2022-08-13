from typing import Optional, Union

import numpy as np

from src.NeuralNet.activation.Activation import Activation


class Sigmoid(Activation):
    def forward(self, inputs: np.array) -> Optional[Union[float, np.array]]:
        return 1 / (1 + np.exp(-inputs))
