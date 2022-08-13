import numpy as np

from src.NeuralNet.loss.Loss import Loss


class CatCrossEntropy(Loss):
    @staticmethod
    def _forward(y_pred: np.array, y_true: np.array):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        # Mask values - only for one-hot encoded labels
        # elif len(y_true.shape) == 2:
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


if __name__ == '__main__':
    print(CatCrossEntropy.calculate([0.7, 0.1, 0.2], [0.0]))
