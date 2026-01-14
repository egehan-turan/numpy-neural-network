import numpy as np
from . import losses
from . import optimizers

class Sequential:
    def __init__(self, layers, loss='MSE', optimizer='SGD', 
                loss_params=None, optimizer_params=None):
        self.layers = layers

        self._loss_params = loss_params or {}
        self._optimizer_params = optimizer_params or {}

        self.loss = loss
        self.optimizer = optimizer

    # --------------------------
    # Loss Property
    # --------------------------
    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, value):
        """
        If value is a string, look it up in the losses module.
        If it's already an object/instance, just assign it.
        """
        if isinstance(value, str):
            try:
                loss_class = getattr(losses, value)
                self._loss = loss_class(**self._loss_params)
            except AttributeError:
                raise ValueError(f"Loss '{value}' not found in losses.py")
        else:
            self._loss = value

    # --------------------------
    # Optimizer Property
    # --------------------------
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        """
        If value is a string, look it up in the losses module.
        If it's already an object/instance, just assign it.
        """
        if isinstance(value, str):
            try:
                opt_class = getattr(optimizers, value)
                self._optimizer = opt_class(**self._optimizer_params)
            except AttributeError:
                raise ValueError(f"Optimizer '{value}' not found in optimizers.py")
        else:
            self._optimizer = value

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            if not layer.built:
                layer.build(X.shape)

            X = layer.forward(X)

        return X

    def backward(self, dY: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dY = layer.backward(dY)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        n_samples = X.shape[-1]

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]

            for i in range(0, n_samples, batch_size):
                X_mini = X_shuffled[:, i : i + batch_size]
                Y_mini = Y_shuffled[:, i : i + batch_size]

                Y_hat = self.forward(X_mini)
                grad = self.loss.gradient(Y_mini, Y_hat)
                self.backward(grad)

                self.optimizer.optimize(self.layers)

                # Reset gradients
                for layer in self.layers:
                    for param in layer.get_parameters():
                        param.zero_grad()

            full_pred = self.forward(X)
            print(f"Epoch {epoch}: Loss = {self.loss.loss(Y, full_pred)}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        Y_hat = self.forward(X)
        
        # If the loss class specifies an activation, use it
        if hasattr(self.loss, 'activation'):
            return self.loss.activation.forward(Y_hat)
            
        return Y_hat
