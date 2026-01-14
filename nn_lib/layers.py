import numpy as np
from .base import Layer
from .parameter import Parameter

class Dense(Layer):
    def __init__(self, output_dim: int, input_dim: int = 0, use_bias: bool = True) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.use_bias = use_bias
        if input_dim != 0:
            self.build([input_dim])

    def build(self, input_shape: int) -> None:
        # input_shape is (features, batch_size)
        self.input_dim = input_shape[0]

        # Xavier/Glorot Initialization
        limit = np.sqrt(1 / self.input_dim)

        W = np.random.randn(self.output_dim, self.input_dim) * limit
        self.W = Parameter(W)

        if self.use_bias:
            B = np.zeros((self.output_dim, 1))
            self.B = Parameter(B)

        self.built = True

    def forward(self, A: np.ndarray) -> np.ndarray:
        self.A = A
        if self.use_bias:
            return (self.W.data @ A) + self.B.data
        else:
            return (self.W.data @ A)

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        self.W.grad += (dZ @ self.A.T) 

        if self.use_bias:
            self.B.grad += np.sum(dZ, axis = -1, keepdims = True) 

        return (self.W.data.T @ dZ)

    def get_parameters(self):
        yield self.W
        if self.use_bias:
            yield self.B