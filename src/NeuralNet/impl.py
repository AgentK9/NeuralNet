from nnfs.datasets import spiral_data

from Network import Network
from activation_functions import rec_linear_func

X, y = spiral_data(100, 3)

n = Network([2, 10, 10, 3], act_func=rec_linear_func)
print(X)
print(n.batch_forward(X))
