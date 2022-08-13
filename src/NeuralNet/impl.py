import numpy as np
from nnfs import init
from nnfs.datasets import spiral_data

from layer import Dense
from activation import RectangularLinear, Softmax
from loss import CatCrossEntropy
from accuracy import Accuracy

init()

X, y = spiral_data(100, 3)
l1 = Dense(2, 3)
a1 = RectangularLinear()
l2 = Dense(3, 3)
a2 = Softmax()

# Layers return raw output of weights/biases
o1 = l1.forward(X)
# Activation Functions return the raw output weighted in some way
o2 = a1.forward(o1)
o3 = l2.forward(o2)
o4 = a2.forward(o3)

# how wrong the network is
loss = CatCrossEntropy.calculate(o4, y)
accuracy = Accuracy.calculate(o4, y)

print(o4[:5])
print("Loss:", loss)
print("Accuracy:", accuracy)
"""
Expected Output:
[[0.33333334 0.33333334 0.33333334]
 [0.33333316 0.3333332  0.33333364]
 [0.33333287 0.3333329  0.33333418]
 [0.3333326  0.33333263 0.33333477]
 [0.33333233 0.3333324  0.33333528]]
Loss: 1.0986104
Accuracy: 0.34
"""
