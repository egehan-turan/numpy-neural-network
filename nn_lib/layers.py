import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view
from .base import Layer
from .parameter import Parameter

class Dense(Layer):
    def __init__(self, 
        output_dim: int, 
        input_dim: int = 0, 
        use_bias: bool = True,
        dtype: npt.DTypeLike=np.float32, 
        activation: str=None
    ) -> None:
        super().__init__(dtype=dtype, activation=activation)
        self.output_dim = output_dim
        self.use_bias = use_bias
        if input_dim != 0:
            self.build([input_dim])

    def build(self, input_shape: tuple[int,int]) -> None:
        # input_shape is (features, batch_size)
        self.input_dim = input_shape[0]

        # Xavier/Glorot Initialization
        limit = np.sqrt(1 / self.input_dim)

        W = np.random.randn(self.output_dim, self.input_dim).astype(self.dtype) * limit
        self.W = Parameter(W, dtype=self.dtype)

        if self.use_bias:
            B = np.zeros((self.output_dim, 1), dtype=self.dtype)
            self.B = Parameter(B, dtype=self.dtype)

        self.built = True

    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)
        self.A = A

        Z = self.W.data @ A
        if self.use_bias:
            Z += self.B.data
        return Z

    def _backward(self, dZ: np.ndarray) -> np.ndarray:
        dZ = self._cast_input(dZ)

        self.W.grad += (dZ @ self.A.T) 

        if self.use_bias:
            self.B.grad += np.sum(dZ, axis = -1, keepdims = True) 

        return (self.W.data.T @ dZ)

    def get_parameters(self):
        yield self.W
        if self.use_bias:
            yield self.B
    
    def get_config(self):
        return {
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'use_bias': self.use_bias,
            'dtype': self.dtype
        }

    def set_parameters(self, weights):
        self.W = Parameter(weights[0])
        if self.use_bias and len(weights) > 1:
            self.B = Parameter(weights[1])
        self.built = True


class Conv2D(Layer):
    def __init__(self, 
        n_filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int,int]=(1, 1),
        padding: bool=True,
        use_bias: bool=True,
        n_channels: int=None,
        dtype: npt.DTypeLike=np.float32,
        activation:str =None
    ) -> None:
        super().__init__(dtype=dtype, activation=activation)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.n_channels = n_channels

    def build(self, input_shape: tuple[int,int,int,int]) -> None:
        if self.n_channels == None:
            self.n_channels = input_shape[0]

        limit = np.sqrt(2 / (self.n_channels * self.kernel_size[0] * self.kernel_size[1]))
        W = np.random.randn(self.n_filters, self.n_channels, *self.kernel_size).astype(self.dtype) * limit
        self.W = Parameter(W, dtype=self.dtype)

        if self.use_bias:
            B = np.zeros((self.n_filters, 1, 1, 1), dtype=self.dtype)
            self.B = Parameter(B, dtype=self.dtype)

        self.built = True
    
    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)
        self.A_shape = A.shape

        if self.padding:
            h_p = (self.kernel_size[0] - 1) // 2
            w_p = (self.kernel_size[1] - 1) // 2
            A = np.pad(A, ((0,0), (h_p,h_p), (w_p,w_p), (0,0)), mode='constant', constant_values=0)
        
        self.A = A
        windows = sliding_window_view(self.A, self.kernel_size, axis=(1, 2)) # Size (C, H, W, N, K1, K2)
        windows = windows[:, ::self.strides[0], ::self.strides[1], ...]

        Z = np.einsum('chwnij,fcij->fhwn', windows, self.W.data, optimize=True)

        if self.use_bias:
            Z += self.B.data

        return Z

    def _backward(self, dZ: np.ndarray) -> np.ndarray:
        dZ = self._cast_input(dZ)

        windows = sliding_window_view(self.A, self.kernel_size, axis=(1, 2))        
        windows = windows[:, ::self.strides[0], ::self.strides[1], ...]

        self.W.grad += np.einsum('fhwn,chwnij->fcij', dZ, windows, optimize=True)

        if self.use_bias:
            self.B.grad += np.sum(dZ, axis=(1,2,3), keepdims = True) 

        _, H_out, W_out, N = dZ.shape
        
        Sh, Sw = self.strides
        dZ_dil = np.zeros(
            (self.n_filters, (H_out - 1) * Sh + 1, (W_out - 1) * Sw + 1, N),
            dtype=self.dtype)
        dZ_dil[:, ::Sh, ::Sw, :] = dZ

        p_h, p_w = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        dZ_padded = np.pad(dZ_dil, ((0,0), (p_h, p_h), (p_w, p_w), (0,0)), mode='constant')

        W_flipped = np.flip(self.W.data, axis=(2, 3))

        dZ_windows = sliding_window_view(dZ_padded, self.kernel_size, axis=(1, 2))
        dX = np.einsum('fhwnij,fcij->chwn', dZ_windows, W_flipped, optimize=True)

        # To handle padding
        diff_h = (dX.shape[1] - self.A_shape[1]) // 2
        diff_w = (dX.shape[2] - self.A_shape[2]) // 2
        
        return dX[:, diff_h : diff_h + self.A_shape[1], diff_w : diff_w + self.A_shape[2], :]

    def get_parameters(self):
        yield self.W
        if self.use_bias:
            yield self.B
    
    def get_config(self):
        return {
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'n_channels': self.n_channels,
            'dtype': self.dtype
        }

    def set_parameters(self, weights):
        self.W = Parameter(weights[0])
        if self.use_bias and len(weights) > 1:
            self.B = Parameter(weights[1])
        self.built = True


class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.built = True

    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)
        self.input_shape = A.shape 
        return A.reshape(-1, self.input_shape[-1])

    def _backward(self, dY: np.ndarray) -> np.ndarray:
        return dY.reshape(self.input_shape)


class MaxPooling2D(Layer):
    def __init__(self, pool_size: tuple[int,int], strides: tuple[int,int] = None) -> None:
        super().__init__()
        self.pool_size = pool_size
        self.built = True

        if strides == None:
            self.strides = pool_size
        else:
            self.strides = pool_size

    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)
        self.A_shape = A.shape
        windows = sliding_window_view(A, self.pool_size, axis=(1, 2))        
        windows = windows[:, ::self.strides[0], ::self.strides[1], ...]
        res = np.max(windows, axis=(-2,-1))
        self.mask = np.argmax(windows.reshape(*windows.shape[:-2], -1), axis=-1)
        return res

    def _backward(self, dZ: np.ndarray) -> np.ndarray:
        n_channels, height, width, n_samples = dZ.shape
        
        grid_h = np.arange(height)[None, :, None, None]
        grid_w = np.arange(width)[None, None, :, None]

        global_h = grid_h * self.strides[0] + self.mask // self.pool_size[1]
        global_w = grid_w * self.strides[1] + self.mask % self.pool_size[1]
        
        dX = np.zeros(self.A_shape)
        
        c_idx = np.arange(n_channels)[:, None, None, None]
        n_idx = np.arange(n_samples)[None, None, None, :]
        
        dX[c_idx, global_h, global_w, n_idx] = dZ
        
        return dX
    
    def get_config(self):
        return {
            'pool_size': self.pool_size,
            'strides': self.strides
        }

class Dropout(Layer):
    def __init__(self, rate: float=0.2):
        super().__init__()
        self.built = True

        self.rate = rate
        self.mask = None
        self.training = True  

    def _forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0:
            return inputs
        
        # Create a binary mask: 1 with probability (1 - rate), 0 with probability rate
        # We use 'Inverted Dropout' by dividing by (1 - rate) so that the expected 
        # sum of activations remains the same during training and testing.
        keep_prob = 1 - self.rate
        self.mask = np.random.binomial(1, keep_prob, size=inputs.shape) / keep_prob
        
        return inputs * self.mask

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Only the neurons that were 'active' in the _forward pass 
        # allow the gradient to flow back.
        return grad_output * self.mask
    
    def get_config(self):
        return {
            'rate': self.rate
        }