import nnfs
from nnfs.datasets import spiral_data

from Network import Network
from src.NeuralNet.old.activation_functions import rec_linear_func, softmax_func
from src.NeuralNet.loss.CatCrossEntropy import CatCrossEntropy

nnfs.init()

X, y = spiral_data(100, 3)

n = Network([(2, None), (3, rec_linear_func), (3, softmax_func)], CatCrossEntropy)
# l1 = Layer(2, 3)
# l2 = Layer(3, 3, softmax_func)
# o1 = l1.forward(X)
# o2 = l2.forward(o1)
out = n.batch_forward(X)[:5]
loss = n.calculate_loss(y)
acc = n.calculate_accuracy(y)
print(out)
print("Loss:", loss)
print("Accuracy: ", acc)
# print(o1[:5])
#print(n.optimize_randomly(10000, x=X, y=y))
