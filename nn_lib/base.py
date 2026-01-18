import numpy as np

class Layer:
    def __init__(self, dtype=None, activation=None):
        self.dtype = dtype
        self.activation = activation
        self.built = False

    # --------------------------
    # Activation Property
    # --------------------------
    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, value):
        """
        If value is a string, look it up in the losses module.
        If it's already an object/instance, just assign it.
        """
        if isinstance(value, str):
            from . import activations
            try:
                activation_class = getattr(activations, value)
                self._activation = activation_class()
                self._activation.dtype = self.dtype
            except AttributeError:
                raise ValueError(f"Activation '{value}' not found in activations.py")
        else:
            self._activation = value
            if value != None:
                self._activation.dtype = self.dtype

    def build(self, input_shape):
        """
        To build the layers lazily. 
        Users are not forced to provide the shape of the input.
        """
        pass

    def _cast_input(self, A):
        """
        Cast input to layer dtype if explicitly set.
        Otherwise inherit dtype from input.
        """
        if self.dtype is not None:
            return np.asarray(A, dtype=self.dtype)
        else:
            self.dtype = A.dtype
            return A
            
    def forward(self, A: np.ndarray) -> np.ndarray:
        """
        Computes the output of the layer for a given input 
        combined with activation.
        """
        Z = self._forward(A)
        
        if self.activation:
            return self.activation.forward(Z)
        return Z

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input
        combined with activation.
        """
        if self.activation:
            dA = self.activation.backward(dA)
        
        # 2. Backprop through the layer's specific math
        return self._backward(dA)

    def _forward(self, x):
        """
        Computes the output of the layer for a given input
        """
        raise NotImplementedError

    def _backward(self, grad):
        """
        Computes the gradient of the loss with respect to the input
        combined with activation.
        """
        raise NotImplementedError

    def get_parameters(self):
        """
        Returns parameters and the gradients of those parameters.
        Form: (variables, current gradients, velocities)
        """
        return
        yield 

    def get_config(self):
        """
        Returns hyperparameters of the layer.
        Form: dict {keyword = value}
        """
        return {}

    def set_parameters(self, weights):
        return

    def __str__(self):
        """
        Returns the name of the class. 
        'Layer', 'Dense', etc.
        """
        return self.__class__.__name__

    def __repr__(self):
        """
        Returns the class name for debugging.
        """
        return f"<{self.__class__.__name__} object>"