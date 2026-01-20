import numpy as np
from numpy.lib.stride_tricks import sliding_window_view, as_strided
from .base import Layer
from .parameter import Parameter

# For type hinting
from collections.abc import Iterator
import numpy.typing as npt

class Dense(Layer):
    """
    Fully connected layer.
    Uses 2D inputs having shape (features, samples).
    """
    def __init__(self, 
        output_dim: int, 
        input_dim: int = 0, 
        use_bias: bool = True,
        dtype: npt.DTypeLike = np.float32, 
        activation: str=None
    ) -> None:
        super().__init__(dtype=dtype, activation=activation)
        self.output_dim = output_dim
        self.use_bias = use_bias
        
        # Build the layer if shape of the input is given
        # TODO: Add this functionality to other layers too.
        if input_dim != 0:
            self.build([input_dim])

    def build(self, input_shape: tuple[int,int]) -> None:
        # Shape is (features, batch_size)
        self.input_dim = input_shape[0]

        # Xavier/Glorot Initialization
        limit = np.sqrt(1 / self.input_dim)

        W = np.random.randn(self.output_dim, self.input_dim).astype(self.dtype) * limit
        self.W = Parameter(W, dtype=self.dtype)

        if self.use_bias:
            B = np.zeros((self.output_dim, 1), dtype=self.dtype)
            self.B = Parameter(B, dtype=self.dtype)

        # Flag as built
        self.built = True

    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)

        # Save A for backward pass
        self.A = A

        # Calculate next layer
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

    def get_parameters(self) -> Iterator[np.ndarray]:
        yield self.W
        if self.use_bias:
            yield self.B
    
    def get_config(self) -> dict:
        return {
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'use_bias': self.use_bias,
            'dtype': self.dtype, 
            'activation': None if self._activation is None else self._activation.__class__.__name__
        }

    def set_parameters(self, weights: list[np.ndarray]) -> None:
        self.W = Parameter(weights[0])

        if self.use_bias and len(weights) > 1:
            self.B = Parameter(weights[1])

        # Flag as built
        self.built = True


class Conv2D(Layer):
    """
    2D Convolutional layer.
    Uses 4D inputs having shape 
    (channels, height, width, samples).
    """
    def __init__(self, 
        n_filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int,int] = (1, 1),
        padding: bool = True,
        use_bias: bool = True,
        n_channels: int = None,
        dtype: npt.DTypeLike = np.float32,
        activation:str = None
    ) -> None:
        super().__init__(dtype=dtype, activation=activation)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

    def build(self, input_shape: tuple[int,int,int,int]) -> None:
        # Shape is (channels, height, width, samples)
        self.n_channels = input_shape[0]

        # Xavier/Glorot Initialization
        limit = np.sqrt(2 / (self.n_channels * self.kernel_size[0] * self.kernel_size[1]))

        W = np.random.randn(self.n_filters, self.n_channels, *self.kernel_size).astype(self.dtype) * limit
        self.W = Parameter(W, dtype=self.dtype)

        if self.use_bias:
            B = np.zeros((self.n_filters, 1, 1, 1), dtype=self.dtype)
            self.B = Parameter(B, dtype=self.dtype)

        # Flag as built
        self.built = True
    
    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)

        # Save original shape of A for backward pass
        self.A_shape = A.shape

        # Pad the image
        if self.padding:
            h_p = (self.kernel_size[0] - 1) // 2
            w_p = (self.kernel_size[1] - 1) // 2
            A = np.pad(A, ((0,0), (h_p,h_p), (w_p,w_p), (0,0)), mode='constant', constant_values=0)
        
        # Save padded A for backward pass
        self.A = A

        # Create windows for convolution and apply strides
        windows = sliding_window_view(self.A, self.kernel_size, axis=(1, 2)) # Size (C, H, W, N, K1, K2)
        windows = windows[:, ::self.strides[0], ::self.strides[1], ...]

        # Calculate the convolution
        Z = np.einsum('chwnij,fcij->fhwn', windows, self.W.data, optimize=True)
        if self.use_bias:
            Z += self.B.data

        return Z

    def _backward(self, dZ: np.ndarray) -> np.ndarray:
        dZ = self._cast_input(dZ)

        # Create windows for backward convolution and apply strides
        windows = sliding_window_view(self.A, self.kernel_size, axis=(1, 2))        
        windows = windows[:, ::self.strides[0], ::self.strides[1], ...]

        self.W.grad += np.einsum('fhwn,chwnij->fcij', dZ, windows, optimize=True)

        if self.use_bias:
            self.B.grad += np.sum(dZ, axis=(1,2,3), keepdims = True) 

        # ----- Calculate gradient of A -----
        _, H_out, W_out, N = dZ.shape
        
        # Dilute dZ with zeros
        # Only relevant if strides are not 1
        Sh, Sw = self.strides
        dZ_dil = np.zeros(
            (self.n_filters, (H_out - 1) * Sh + 1, (W_out - 1) * Sw + 1, N),
            dtype=self.dtype)
        dZ_dil[:, ::Sh, ::Sw, :] = dZ

        # Pad dZ
        p_h, p_w = self.kernel_size[0] - 1, self.kernel_size[1] - 1
        dZ_dil = np.pad(dZ_dil, ((0,0), (p_h, p_h), (p_w, p_w), (0,0)), mode='constant')

        # Flip W and calculate the convolution
        W_flipped = np.flip(self.W.data, axis=(2, 3))
        dZ_windows = sliding_window_view(dZ_dil, self.kernel_size, axis=(1, 2))
        dX = np.einsum('fhwnij,fcij->chwn', dZ_windows, W_flipped, optimize=True)

        # To handle padding in forward layer
        diff_h = (dX.shape[1] - self.A_shape[1]) // 2
        diff_w = (dX.shape[2] - self.A_shape[2]) // 2
        
        return dX[:, diff_h : diff_h + self.A_shape[1], diff_w : diff_w + self.A_shape[2], :]

    def get_parameters(self) -> Iterator[np.ndarray]:
        yield self.W
        if self.use_bias:
            yield self.B
    
    def get_config(self) -> dict:
        return {
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'n_channels': self.n_channels,
            'dtype': self.dtype, 
            'activation': None if self._activation is None else self._activation.__class__.__name__
        }

    def set_parameters(self, weights: list[np.ndarray]) -> None:
        self.W = Parameter(weights[0])

        if self.use_bias and len(weights) > 1:
            self.B = Parameter(weights[1])

        self.built = True


class Flatten(Layer):
    """
    Flattens tensor from any shape to (features, batch_size).
    """
    def __init__(self) -> None:
        super().__init__()

        # Flag as built since this layer has no parameters.
        self.built = True

    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)
        self.input_shape = A.shape 
        return A.reshape(-1, self.input_shape[-1])

    def _backward(self, dY: np.ndarray) -> np.ndarray:
        return dY.reshape(self.input_shape)


class MaxPooling2D(Layer):
    """
    2D MaxPooling layer.
    Uses 4D inputs having shape 
    (channels, height, width, samples).
    """
    def __init__(self, 
        pool_size: tuple[int,int], 
        strides: tuple[int,int] = None
    ) -> None:
        super().__init__()
        self.pool_size = pool_size

        # Flag as built since this layer has no parameters.
        self.built = True

        # If no strides are provided apply it in an non-overlapping fashion
        if strides == None:
            self.strides = pool_size
        else:
            self.strides = strides

    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)

        # Save original shape of A for backward pass
        self.A_shape = A.shape

        # Create windows for convolution and apply strides
        windows = sliding_window_view(A, self.pool_size, axis=(1, 2))        
        windows = windows[:, ::self.strides[0], ::self.strides[1], ...]

        # Take maximum in every window
        res = np.max(windows, axis=(-2,-1))

        # Save which indicies are chosen for backward pass
        self.mask = np.argmax(windows.reshape(*windows.shape[:-2], -1), axis=-1)

        return res

    def _backward(self, dZ: np.ndarray) -> np.ndarray:
        """
        Place dZ to appopriate places in the original matrix
        """
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
    
    def get_config(self) -> dict:
        return {
            'pool_size': self.pool_size,
            'strides': self.strides
        }


class Dropout(Layer):
    """
    Dropout layer for regularization. 
    Randomly drops rate of units during training.
    Inputs can have any shape.
    """
    def __init__(self, 
        rate: float = 0.2
    ) -> None:
        super().__init__()

        self.rate = rate
        self.mask = None

        # Flag to determine mode (training, prediction)
        self.training = True  

        # Flag as built since this layer has no parameters.
        self.built = True

    def _forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.training or self.rate == 0:
            return inputs
        
        # Create a binary mask: 1 with prob (1 - rate), 0 with prob (rate)
        # Using Inverted Dropout
        keep_prob = 1 - self.rate

        # Save which neurons was active for backward pass
        self.mask = np.random.binomial(1, keep_prob, size=inputs.shape) / keep_prob
        
        return inputs * self.mask

    def _backward(self, grad_output: np.ndarray) -> np.ndarray:
        # Only the neurons that were active in the forward pass 
        return grad_output * self.mask
    
    def get_config(self) -> dict:
        return {
            'rate': self.rate
        }


class BatchNorm(Layer):
    """
    Batch normalization layer.
    Uses M-D inputs having shape (feature_axes, samples)
    """
    def __init__(self, 
        momentum: float = 0.99,
        epsilon: float = 0.001,
        running_mean: float = None,
        running_var: float = None,
        dtype: npt.DTypeLike = np.float32, 
        activation: str = None
    ) -> None:
        super().__init__(dtype=dtype, activation=activation)
        self.momentum = momentum
        self.epsilon = epsilon

        # Flag to determine mode (training, prediction)
        self.training = True

        if running_mean is not None:
            self.running_mean = running_mean
            self.running_var = running_var

    def build(self, input_shape: tuple[int]) -> None:
        gamma = np.ones((*input_shape[:-1], 1), dtype=self.dtype)
        self.gamma = Parameter(gamma, dtype=self.dtype)

        beta = np.zeros((*input_shape[:-1], 1), dtype=self.dtype)
        self.beta = Parameter(beta, dtype=self.dtype)

        # Keep overall mean and variance for predictions
        self.running_mean = np.zeros((*input_shape[:-1], 1), dtype=self.dtype)
        self.running_var = np.ones((*input_shape[:-1], 1), dtype=self.dtype)

        # Flag as built
        self.built = True

    def _forward(self, X: np.ndarray) -> np.ndarray:
        X = self._cast_input(X)
        
        if self.training:
            # If training determine current mean and variance
            mu = np.mean(X, axis=-1, keepdims=True)
            var = np.var(X, axis=-1, keepdims=True)
            
            # Update running mean and average
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            # If prediction use overall mean and variance
            mu = self.running_mean
            var = self.running_var

        # Keep 1 / (standart deviation) and (X - mean) for backward pass
        self.std_inv = 1.0 / np.sqrt(var + self.epsilon)
        self.X_centered = X - mu

        # Calculated standardized X
        self.X_hat = self.X_centered * self.std_inv
        
        return self.gamma.data * self.X_hat + self.beta.data

    def _backward(self, dY: np.ndarray) -> np.ndarray:
        dY = self._cast_input(dY)
        n_samples = dY.shape[-1] 
  
        self.gamma.grad += np.sum(dY * self.X_hat, axis=-1, keepdims=True)
        self.beta.grad += np.sum(dY, axis=-1, keepdims=True)
        
        dX_hat = dY * self.gamma.data
        
        # Batch norm gradient formula
        dX = (1. / n_samples) * self.std_inv * (
            n_samples * dX_hat - 
            np.sum(dX_hat, axis=-1, keepdims=True) - 
            self.X_hat * np.sum(dX_hat * self.X_hat, axis=-1, keepdims=True)
        )
        
        return dX

    def get_parameters(self) -> Iterator[np.ndarray]:
        yield self.gamma
        yield self.beta
    
    def get_config(self) -> dict:
        return {
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'running_mean': self.running_mean,
            'running_var': self.running_var,
            'dtype': self.dtype, 
            'activation': None if self._activation is None else self._activation.__class__.__name__
        }

    def set_parameters(self, weights: list[np.ndarray]) -> None:
        self.gamma = Parameter(weights[0])
        self.beta = Parameter(weights[1])

        # Flag as built
        self.built = True


class FastConv2D(Layer):
    """
    Optimized 2D convolutional layer using im2col for efficiency.
    Uses 4D inputs having shape 
    (channels, height, width, samples).
    """
    def __init__(self, 
        n_filters: int,
        kernel_size: tuple[int, int],
        strides: tuple[int,int] = (1, 1),
        padding: bool = True,
        use_bias: bool = True,
        n_channels: int = None,
        dtype: npt.DTypeLike = np.float32,
        activation: str = None
    ) -> None:
        super().__init__(dtype=dtype, activation=activation)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias

    def build(self, input_shape: tuple[int,int,int,int]) -> None:
        # Shape is (channels, height, width, samples)
        self.n_channels = input_shape[0]

        # Xavier/Glorot Initialization
        limit = np.sqrt(2 / (self.n_channels * self.kernel_size[0] * self.kernel_size[1]))

        W = np.random.randn(self.n_filters, 
            self.n_channels * self.kernel_size[0] * self.kernel_size[1]).astype(self.dtype) * limit
        self.W = Parameter(W, dtype=self.dtype)

        if self.use_bias:
            B = np.zeros((self.n_filters, 1), dtype=self.dtype)
            self.B = Parameter(B, dtype=self.dtype)

        # Flag as built
        self.built = True
    
    def _im2col(self, A):
        """
        Fast im2col using stride tricks.
        """
        C, H, W, N = A.shape
        Kh, Kw = self.kernel_size
        Sh, Sw = self.strides
        
        H_out = (H - Kh) // Sh + 1
        W_out = (W - Kw) // Sw + 1
        
        s0, s1, s2, s3 = A.strides
        
        shape = (C, Kh, Kw, H_out, W_out, N)
        strides = (s0, s1, s2, s1*Sh, s2*Sw, s3)
        
        windows = as_strided(A, shape=shape, strides=strides)
        
        col = windows.transpose(3, 4, 5, 0, 1, 2).reshape(H_out * W_out * N, -1).T
        
        return col, H_out, W_out
        
    def _col2im(self, col, input_shape, output_shape):
        """
        Convert column matrix back to image.
        """
        C, H, W, N = input_shape
        H_out, W_out = output_shape
        Kh, Kw = self.kernel_size
        Sh, Sw = self.strides
        
        # Fast path: (stride == kernel_size)
        if Sh == Kh and Sw == Kw and H == H_out * Kh and W == W_out * Kw:
            col = col.reshape(C, Kh, Kw, H_out, W_out, N)
            col = col.transpose(0, 3, 1, 4, 2, 5)
            return col.reshape(C, H, W, N)
        
        col = col.reshape(C, Kh, Kw, H_out, W_out, N)
        dA = np.zeros(input_shape, dtype=self.dtype)
        
        for kh in range(Kh):
            for kw in range(Kw):
                h_start = kh
                h_end = h_start + H_out * Sh
                w_start = kw
                w_end = w_start + W_out * Sw
                
                dA[:, h_start:h_end:Sh, w_start:w_end:Sw, :] += col[:, kh, kw, :, :, :]
        
        return dA
    
    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)
        self.A_shape = A.shape

        if self.padding:
            h_p = (self.kernel_size[0] - 1) // 2
            w_p = (self.kernel_size[1] - 1) // 2
            A = np.pad(A, ((0,0), (h_p,h_p), (w_p,w_p), (0,0)), mode='constant')
        
        self.A = A
        
        col, H_out, W_out = self._im2col(A)
        self.col = col
        self.output_shape = (H_out, W_out)
        
        Z = self.W.data @ col  

        if self.use_bias:
            Z += self.B.data
        
        Z = Z.reshape(self.n_filters, H_out, W_out, -1)

        return Z

    def _backward(self, dZ: np.ndarray) -> np.ndarray:
        dZ = self._cast_input(dZ)
        
        F, _, _, N = dZ.shape
        
        # Reshape dZ for matrix operations
        dZ_col = dZ.reshape(F, -1) 
        
        dW = dZ_col @ self.col.T  
        self.W.grad += dW
        
        if self.use_bias:
            self.B.grad += np.sum(dZ_col, axis=1, keepdims=True)
        
        dA_col = self.W.data.T @ dZ_col  
        
        dA = self._col2im(dA_col, self.A.shape, self.output_shape)
        
        if self.padding:
            diff_h = (dA.shape[1] - self.A_shape[1]) // 2
            diff_w = (dA.shape[2] - self.A_shape[2]) // 2
            dA = dA[:, diff_h:diff_h + self.A_shape[1], diff_w:diff_w + self.A_shape[2], :]
        
        return dA

    def get_parameters(self) -> Iterator[np.ndarray]:
        yield self.W
        if self.use_bias:
            yield self.B
    
    def get_config(self) -> dict:
        return {
            'n_filters': self.n_filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'n_channels': self.n_channels,
            'dtype': self.dtype, 
            'activation': None if self._activation is None else self._activation.__class__.__name__
        }

    def set_parameters(self, weights: list[np.ndarray]) -> None:
        self.W = Parameter(weights[0])

        if self.use_bias and len(weights) > 1:
            self.B = Parameter(weights[1])

        # Flag as built.
        self.built = True


class FastMaxPooling2D(Layer):
    """
    Optimized 2D max pooling layer using im2col for efficiency.
    """
    def __init__(self, 
        pool_size: tuple[int, int] = (2, 2),
        strides: tuple[int, int] = None,
        padding: bool = False,
        dtype: npt.DTypeLike = np.float32
    ) -> None:
        super().__init__(dtype=dtype)
        self.pool_size = pool_size
        self.strides = strides if strides is not None else pool_size
        self.padding = padding
        self.use_fast_path = (self.strides == self.pool_size and not padding)
        self.built = True
    
    def _im2col_pool(self, A):
        C, H, W, N = A.shape
        Ph, Pw = self.pool_size
        Sh, Sw = self.strides
        
        H_out = (H - Ph) // Sh + 1
        W_out = (W - Pw) // Sw + 1
        
        s0, s1, s2, s3 = A.strides
        shape = (C, Ph, Pw, H_out, W_out, N)
        strides = (s0, s1, s2, s1*Sh, s2*Sw, s3)
        
        windows = as_strided(A, shape=shape, strides=strides)
        
        col = windows.transpose(0, 3, 4, 5, 1, 2).reshape(C, H_out*W_out*N, Ph*Pw)
        
        return col, H_out, W_out
    
    def _forward(self, A: np.ndarray) -> np.ndarray:
        A = self._cast_input(A)
        self.A_shape = A.shape
        
        if self.padding:
            Ph, Pw = self.pool_size
            h_p = (Ph - 1) // 2
            w_p = (Pw - 1) // 2
            A = np.pad(A, ((0,0), (h_p,h_p), (w_p,w_p), (0,0)), 
                      mode='constant', constant_values=-np.inf)
        
        self.A = A
        C, H, W, N = A.shape
        Ph, Pw = self.pool_size
        
        # Fast path: non-overlapping pooling
        if self.use_fast_path and H % Ph == 0 and W % Pw == 0:
            A_reshaped = A.reshape(C, H//Ph, Ph, W//Pw, Pw, N)
            Z = A_reshaped.max(axis=(2, 4))
            
            # Store mask for backward pass
            A_reshaped_max = Z[:, :, np.newaxis, :, np.newaxis, :]
            self.mask = (A_reshaped == A_reshaped_max)
            self.output_shape = Z.shape[1:3]
            
            return Z
        
        # Slow path: overlapping pooling or non-divisible sizes
        col, H_out, W_out = self._im2col_pool(A)
        self.col_shape = col.shape
        self.output_shape = (H_out, W_out)
        
        Z = col.max(axis=-1)  
        
        # Create mask for backward pass
        max_vals = Z[:, :, np.newaxis]  
        self.mask = (col == max_vals)  
        
        self.mask = self.mask / self.mask.sum(axis=-1, keepdims=True)
        
        Z = Z.reshape(C, H_out, W_out, N)
        
        return Z
    
    def _backward(self, dZ: np.ndarray) -> np.ndarray:
        dZ = self._cast_input(dZ)
        
        C, H_out, W_out, N = dZ.shape
        Ph, Pw = self.pool_size
        
        # Fast path backward
        if self.use_fast_path and self.A.shape[1] % Ph == 0 and self.A.shape[2] % Pw == 0:
            # Expand gradient back to input shape
            dZ_expanded = dZ[:, :, np.newaxis, :, np.newaxis, :]
            dA = (dZ_expanded * self.mask).reshape(self.A.shape)
            
        else:
            # Slow path backward using mask
            dZ_flat = dZ.reshape(C, H_out*W_out*N)
            
            dZ_col = dZ_flat[:, :, np.newaxis] * self.mask  
            
            dZ_col = dZ_col.reshape(C, H_out, W_out, N, Ph, Pw)
            
            dA = self._col2im_pool(dZ_col, self.A.shape)
        
        # Remove padding 
        if self.padding:
            diff_h = (dA.shape[1] - self.A_shape[1]) // 2
            diff_w = (dA.shape[2] - self.A_shape[2]) // 2
            dA = dA[:, diff_h:diff_h + self.A_shape[1], 
                      diff_w:diff_w + self.A_shape[2], :]
        
        return dA
    
    def get_config(self):
        return {
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
            'dtype': self.dtype
        }