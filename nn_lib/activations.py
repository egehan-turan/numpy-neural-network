import numpy as np 
from .base import Layer

class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.built = True

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA: np.ndarray) -> np.ndarray:
        self.dZ = dA.copy()
        self.dZ[self.Z <= 0] = 0
        return self.dZ


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.built = True

    def forward(self, Z: np.ndarray) -> np.ndarray:
        self.A = 1 / (1 + np.exp(-Z))
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        return dA * self.A * (1 - self.A)

class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.built = True

    def forward(self, Z: np.ndarray) -> np.ndarray:
        # For this to work in arbitrary dimentions
        feature_axes = tuple(range(Z.ndim - 1))
    
        Z_normalized = Z - np.max(Z, axis=feature_axes, keepdims=True)

        exps = np.exp(Z_normalized)
        self.A = exps / np.sum(exps, axis=feature_axes, keepdims=True)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Jacobian is A_i * (delta_ij - A_j)
        Distributing we get dZ_i = A_i * (dA_i - sum_j(dA_j * A_j))
        """
        feature_axes = tuple(range(dA.ndim - 1))

        # Used in calculation of jacobian
        sum_weighted_dA = np.sum(dA * self.A, axis=feature_axes, keepdims=True)

        return self.A * (dA - sum_weighted_dA)