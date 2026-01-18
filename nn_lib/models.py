import numpy as np
from . import losses
from . import optimizers
from . import helpers
from tqdm import tqdm
import pickle
import time


class Model:
    def __init__(self, layers, loss, optimizer, 
                loss_params: dict=None, optimizer_params: dict=None,
                sample_axis: int=-1):
        self.layers = layers
        self.sample_axis=sample_axis

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
                optimizer_class = getattr(optimizers, value)
                self._optimizer = optimizer_class(**self._optimizer_params)
            except AttributeError:
                raise ValueError(f"Optimizer '{value}' not found in optimizers.py")
        else:
            self._optimizer = value

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 10, batch_size: int = 32,
        loss_threshold: float=None, update_interval: float=0.5) -> None:
        
        # Toggle training flag
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

        # This framework works in form (features_axes, samples)
        X = np.moveaxis(X, self.sample_axis, -1)
        Y = np.moveaxis(Y, self.sample_axis, -1)
        
        n_samples = X.shape[-1]
        n_batches = (n_samples + batch_size - 1) // batch_size
    
        print(f"Training started: {n_samples} samples, {n_batches} batches per epoch.")
        start_time = time.time()

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[..., indices]
            Y_shuffled = Y[..., indices]

            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(range(0, n_samples, batch_size), 
                        desc=f"Epoch {epoch+1}/{epochs}",
                        unit="batch",
                        mininterval=update_interval)
            for i in pbar:
                X_mini = X_shuffled[..., i : i + batch_size]
                Y_mini = Y_shuffled[..., i : i + batch_size]

                Y_hat = self.forward(X_mini)
                batch_loss = self.loss.loss(Y_mini, Y_hat)
                epoch_loss += batch_loss
                n_batches += 1

                grad = self.loss.gradient(Y_mini, Y_hat)
                self.backward(grad)
                self.optimizer.optimize(self.layers)

                # Reset gradients
                for layer in self.layers:
                    for param in layer.get_parameters():
                        param.zero_grad()

                pbar.set_postfix({'loss': helpers.format_number(epoch_loss / n_batches)}, refresh=False)

            if loss_threshold and epoch_loss / n_batches < loss_threshold:
                print(f"Loss threshold reached: Loss={epoch_loss / n_batches:.6f} <= Threshold={loss_threshold}")
                print("Training finished.")
                break

        results = {
        "training_time": time.time() - start_time,
        "final_loss": epoch_loss / n_batches,
        "epochs_completed": epoch + 1
        }
        
        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Toggle training flag
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False

        X = np.moveaxis(X, self.sample_axis, -1)

        Y_hat = self.forward(X)
        
        # If the loss class specifies an activation, use it
        if hasattr(self.loss, 'activation'):
            Y_hat = self.loss.activation.forward(Y_hat)
            
        Y_hat = np.moveaxis(Y_hat, -1, self.sample_axis)
        return Y_hat

    def save(self, filepath: str) -> None:
        """
        Save the model architecture and parameters to a file.
        Only saves essential information: layer configs and parameter values.
        """
        # Extract lightweight layer information only containing parameters
        layers_data = []
        for layer in self.layers:
            layer_data = {
                'class_name': layer.__class__.__name__,
                'module': layer.__class__.__module__,
                'config': layer.get_config() if hasattr(layer, 'get_config') else {},
                'parameters': [param.data for param in layer.get_parameters()]
            }
            layers_data.append(layer_data)
        
        model_data = {
            'class_name': self.__class__.__name__,
            'layers': layers_data,
            'loss': self._loss.__class__.__name__,
            'optimizer': self._optimizer.__class__.__name__,
            'loss_params': self._loss_params,
            'optimizer_params': self._optimizer_params
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> 'Model':
        """
        Load a model from a file.
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        layers = []
        for layer_data in model_data['layers']:
            # Read layer name and find class
            module_parts = layer_data['module'].split('.')
            if module_parts[0] == '.':
                from . import layers as layers_module

                layer_class = getattr(layers_module, layer_data['class_name'])
            else:
                import importlib
                
                module = importlib.import_module(layer_data['module'])
                layer_class = getattr(module, layer_data['class_name'])
            
            # Create layer instance
            layer = layer_class(**layer_data['config'])
            
            # Set parameters if available using the layer's set_parameters method
            if layer_data['parameters'] and hasattr(layer, 'set_parameters'):
                layer.set_parameters(layer_data['parameters'])
            
            layers.append(layer)
        
        # Read model name and find class
        model_class = globals().get(model_data['class_name'], cls)

        # Create model instance
        model = model_class(
            layers=layers,
            loss=model_data['loss'],
            optimizer=model_data['optimizer'],
            loss_params=model_data['loss_params'],
            optimizer_params=model_data['optimizer_params']
        )
        
        print(f"Model loaded from {filepath}")
        return model


class Sequential(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, X: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            if not layer.built:
                layer.build(X.shape)

            X = layer.forward(X)

        return X

    def backward(self, dY: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dY = layer.backward(dY)