class Layer:
    def __init__(self):
        self.built = False

    def build(self, input_shape):
        """
        To build the layers lazily. 
        Users are not forced to provide the shape of the input.
        """
        pass

    def forward(self, input_data):
        """
        Computes the output of the layer for a given input.
        """
        raise NotImplementedError

    def backward(self, output_gradient):
        """
        Computes the gradient of the loss with respect to the input.
        """
        raise NotImplementedError

    def get_parameters(self):
        """
        Returns parameters and the gradients of those parameters.
        Form: (variables, current gradients, velocities)
        """
        return
        yield 

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