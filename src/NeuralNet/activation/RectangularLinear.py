from typing import Optional, Union

import numpy as np

from src.NeuralNet.activation.Activation import Activation


class RectangularLinear(Activation):
    def forward(self, inputs: np.array) -> Optional[Union[float, np.array]]:
        return np.maximum(0, inputs)
