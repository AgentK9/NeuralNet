import nnfs
import numpy as np
from nnfs.datasets import vertical_data

import layer
import activation
import loss
from accuracy import Accuracy

nnfs.init()

# data
X, y = vertical_data(100, 3)

# network
l1 = layer.Dense(2, 3)
a1 = activation.RectangularLinear()
l2 = layer.Dense(3, 3)
a2 = activation.Softmax()

# loss
loss_fn = loss.CatCrossEntropy()

# accuracy
acc_fn = Accuracy()

# helpers
lowest_loss = np.inf
best_l1_weights = l1.weights.copy()
best_l1_biases = l1.biases.copy()
best_l2_weights = l2.weights.copy()
best_l2_biases = l2.biases.copy()

for i in range(10000):
    l1.randomize_weights()
    l1.randomize_biases()
    l2.randomize_weights()
    l2.randomize_biases()

    o1 = l1.forward(X)
    o2 = a1.forward(o1)
    o3 = l2.forward(o2)
    o4 = a2.forward(o3)

    loss = loss_fn.calculate(o4, y)
    accuracy = acc_fn.calculate(o4, y)

    if loss < lowest_loss:
        print("New set of weights found:")
        print(f"\tIteration: {i}\n\tLoss: {loss}\n\tAccuracy: {accuracy}")
        best_l1_weights = l1.weights.copy()
        best_l1_biases = l1.biases.copy()
        best_l2_weights = l2.weights.copy()
        best_l2_biases = l2.biases.copy()
        lowest_loss = loss
