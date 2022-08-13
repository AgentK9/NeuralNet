from typing import Protocol, Optional, Union

import numpy as np


class Activation(Protocol):
    _output: Optional[Union[float, np.array]]

    def __init__(self):
        self._output = None

    def forward(self, inputs: np.array) -> Optional[Union[float, np.array]]:
        pass

    def get_output(self) -> Optional[Union[float, np.array]]:
        return self._output
