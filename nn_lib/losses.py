"""
Do batch normalizations here and pass the normalized steps.
Hence do not divide by batchsize in layers.
Equivalent to calculating gradient of average ove batch.
"""
import numpy as np
from .activations import Softmax

class MSE:
    def __init__(self) -> None:
        pass

    def loss(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        return np.mean(np.square(Y - Y_hat))

    def gradient(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        return 2 * (Y_hat - Y) / Y.size

class CategoricalCrossEntropy:
    def __init__(self) -> None:
        pass

    def loss(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        n_samples = Y_hat.shape[-1]
        return -np.sum(Y * np.log(Y_hat + 1e-15)) / n_samples

    def gradient(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        n_samples = Y_hat.shape[-1]
        return -(Y / (Y_hat + 1e-15)) / n_samples

class SoftmaxCCE:
    """
    SoftmaxCCE (Fused Softmax and Categorical Cross-Entropy)

    WARNING: This loss function is an optimized 'fused' implementation. 
    Do NOT include a Softmax layer at the end of your model when using this class.
    """
    def __init__(self) -> None:
        self.softmax_layer = Softmax()

    def loss(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        n_samples = Y_hat.shape[-1]
        Y_hat = self.softmax_layer.forward(Y_hat)
        return -np.sum(Y * np.log(Y_hat+ 1e-15)) / n_samples

    def gradient(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        n_samples = Y_hat.shape[-1]
        return (Y_hat - Y) / n_samples