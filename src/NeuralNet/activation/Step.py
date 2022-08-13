from typing import Optional, Union

import numpy as np

from src.NeuralNet.activation.Activation import Activation


class Step(Activation):

    def forward(self, inputs: np.array) -> Optional[Union[float, np.array]]:
        self._output = np.maximum(0, inputs)
        return self._output
