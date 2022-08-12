from typing import List

import numpy as np


def cat_cross_entropy(output: List[float], y: int) -> float:
    return -np.log(output[y])


if __name__ == '__main__':
    print(cat_cross_entropy([0.7, 0.1, 0.2], 0))
