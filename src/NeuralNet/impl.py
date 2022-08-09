import nnfs
from nnfs.datasets import spiral_data

from Network import Network
from activation_functions import rec_linear_func, softmax_func
from Layer import Layer

nnfs.init()

X, y = spiral_data(100, 3)

n = Network([(2, None), (3, rec_linear_func), (3, softmax_func)])
# l1 = Layer(2, 3)
# l2 = Layer(3, 3, softmax_func)
# o1 = l1.forward(X)
# o2 = l2.forward(o1)
y = n.batch_forward(X)[:5]
print(y)
# print(o1[:5])
