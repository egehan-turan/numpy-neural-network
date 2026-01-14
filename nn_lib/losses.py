"""
Do batch normalizations here and pass the normalized steps.
Hence do not divide by batchsize in layers.
Equivalent to calculating gradient of average ove batch.
"""
import numpy as np
from .activations import Sigmoid, Softmax

class MSE:
    def __init__(self) -> None:
        pass

    def loss(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        return np.mean(np.square(Y - Y_hat))

    def gradient(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        return 2 * (Y_hat - Y) / Y.size


class CCE:
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

    WARNING: Do NOT include a Softmax layer at the end of your model when using this class.
    """

    def __init__(self) -> None:
        self.activation = Softmax()

    def loss(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        n_samples = Y_hat.shape[-1]
        Y_hat = self.activation.forward(Y_hat)
        return -np.sum(Y * np.log(Y_hat+ 1e-15)) / n_samples

    def gradient(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        n_samples = Y_hat.shape[-1]
        Y_hat = self.activation.forward(Y_hat)
        return (Y_hat - Y) / n_samples


class BCE:
    def __init__(self) -> None:
        pass

    def loss(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        """
        Y and Y_hat shape: (1, n_samples)
        """
        n_samples = Y_hat.shape[-1]
        return -np.sum(Y * np.log(Y_hat + 1e-15) + (1 - Y) * np.log(1 - Y_hat + 1e-15)) / n_samples

    def gradient(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        n_samples = Y_hat.shape[-1]
        return -(Y / (Y_hat + 1e-15) - (1 - Y) / (1 - Y_hat + 1e-15)) / n_samples


class SigmoidBCE:
    """
    SigmoidBCE (Fused Sigmoid and Binary Cross-Entropy)

    WARNING: Do NOT include a Sigmoid layer at the end of your model when using this class.
    """

    def __init__(self) -> None:
        self.activation = Sigmoid()

    def loss(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.float64:
        """
        Y and Y_hat shape: (1, n_samples)
        """
        n_samples = Y_hat.shape[-1]
        Y_hat = self.activation.forward(Y_hat)
        Y_hat = np.clip(Y_hat, 1e-15, 1.0 - 1e-15)
        return -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) / n_samples

    def gradient(self, Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
        """
        Returns gradient w.r.t the raw pre-activation outputs
        """
        n_samples = Y_hat.shape[-1]
        Y_hat = self.activation.forward(Y_hat)
        return (Y_hat - Y) / n_samples