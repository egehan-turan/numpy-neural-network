# Documentation

NumPy-based neural network library for modular deep learning research and education.

---

## Table of Contents

- [Core Classes](#core-classes)
  - [Base Layer](#layer-basepy)
  - [Dense](#dense)
  - [Conv2D](#conv2d)
  - [Flatten](#flatten)
  - [MaxPooling2D](#maxpooling2d)
  - [Dropout](#dropout)
  - [BatchNorm](#batchnorm)
  - [FastConv2D](#fastconv2d)
  - [FastMaxPooling2D](#fastmaxpooling2d)
- [Activations](#activations)
- [Losses](#losses)
- [Optimizers](#optimizers)
- [Models](#models)
- [Helpers & Parameters](#helpers--parameters)

---

## Core Classes

### Layer (`base.py`)
Base class for all layers.

**Constructor**
```python
Layer(dtype=None, activation=None)
```
- `dtype`: data type (NumPy type)
- `activation`: activation function (string or instance)

**Main Methods:**
- `forward(A)`: Forward pass with activation.
- `backward(dA)`: Backward pass with activation. Stores gradients for optimizers.
- `_forward(A)`: Forward pass without activation.
- `_backward(dA)`: Backward pass without activation.
- `build(input_shape)`: Define params lazily.
- `_cast_input(A)`: Set dtype, cast input.
- `get_parameters()`: Generator for params/gradients.
- `get_config()`: Dict of hyperparameters.
- `__str__()`, `__repr__()`: Class name.

---

### Dense (`layers.py`)
Fully connected layer.

```python
Dense(output_dim, input_dim=0, use_bias=True, dtype=np.float32, activation=None)
```
- `build(input_shape)`
- `_forward(A)`
- `_backward(dZ)`
- `get_parameters()`
- `set_parameters(weights)`
- `get_config()`

---

### Conv2D (`layers.py`)
2D convolutional layer. Input shape: `(channels, height, width, samples)`

```python
Conv2D(n_filters, kernel_size, strides=(1,1), padding=True, use_bias=True, n_channels=None, dtype=np.float32, activation=None)
```
- `build(input_shape)`
- `_forward(A)`  
- `_backward(dZ)`
- `get_parameters()` 
- `set_parameters(weights)`
- `get_config()`

---

### Flatten (`layers.py`)
Flattens tensor for fully connected input.

```python
Flatten()
```
- `_forward(A)`
- `_backward(dY)`

---

### MaxPooling2D (`layers.py`)
2D Max Pooling, for shape `(channels, height, width, samples)`

```python
MaxPooling2D(pool_size, strides=None)
```
- `_forward(A)`
- `_backward(dZ)`
- `get_config()`

---

### Dropout (`layers.py`)
Randomly deactivates neurons during training.

```python
Dropout(rate=0.2)
```
- `_forward(inputs)`
- `_backward(grad_output)`
- `get_config()`

---

### BatchNorm (`layers.py`)
Batch normalization across feature axes.

```python
BatchNorm(momentum=0.99, epsilon=0.001, dtype=np.float32, activation=None)
```
- `build(input_shape)`
- `_forward(X)`
- `_backward(dY)`
- `get_parameters()`
- `set_parameters(weights)`
- `get_config()`

---

### FastConv2D & FastMaxPooling2D (`layers.py`)
Optimized versions (using im2col).

**FastConv2D**: Like Conv2D, but faster via stride tricks.  
**FastMaxPooling2D**: Like MaxPooling2D, optimized for non-overlapping windows.

---

## Activations

Implemented in `activations.py` as subclasses of `Layer`.

- **ReLU**: Rectified linear unit
- **Sigmoid**
- **Softmax**

Each implements:
- `_forward(Z)`: Forward Pass
- `_backward(dA)`: Backward Pass

---

## Losses (`losses.py`)

- **MSE**: Mean squared error (regression)
- **CCE**: Categorical cross-entropy (multi-class)
- **BCE**: Binary cross-entropy
- **SoftmaxCCE**: Softmax + CCE, fused for efficiency (no external Softmax layer)
  If used, the model should NOT include *Softmax* as its last layer. Optimizes the gradient pass process. Recommended instead of using *CCE* and *Softmax* separately.
- **SigmoidBCE**: Sigmoid + BCE, fused for efficiency
  If used, the model should NOT include *Sigmoid* as its last layer. Optimizes the gradient pass process. Recommended instead of using *BCE* and *Sigmoid* separately.

Each implements:
- `loss(Y, Y_hat)`: Calculates loss for given training data and predictions.
- `gradient(Y, Y_hat)`: Calculates gradient for given training data and predictions.

---

## Optimizers (`optimizers.py`)

### **SGD**: Stochastic gradient descent with momentum  
```python
SGD(learning_rate=0.01, momentum=0.9)
```

### **Adam**: Adam optimizer  
```python
Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
```
  
Each implements:
  - `optimize(layers)`: Runs over the layers and changes their parameters.


---

## Models (`models.py`)

### Model
Top-level model/train interface.

```python
Model(layers, loss, optimizer, loss_params=None, optimizer_params=None, sample_axis=-1)
```

- `train(X, Y, epochs=10, batch_size=32, ...)`: Mini-batch training loop
- `predict(X)`: Given input produces prediction
- `save(filepath)`: Saves the model to *filepath* 
- `load(filepath)`: Loads the model from *filepath*

#### Sequential (subclass)
- `forward(X)`: Chain inputs through layers
- `backward(dY)`: Propagate gradients backward

---

## Helpers & Parameters

- **Parameter (`parameter.py`)**: Lightweight class for weights and gradients.
  - `Parameter(data, dtype=np.float32)`
  - `zero_grad()`: Clear gradient
  - `__hash__`, `__eq__` (for optimizer state dicts)
  - `data`, `grad`
  
- **Helpers**: Misc utility functions, e.g. `format_number(x)` for pretty-printing floats.

---

## Notes

- All shapes are assumed to be `(features..., samples)` for fastest broadcasting. For other shapes model *sample_axis* parameter needs to be set.
- Layers and losses can be extended by subclassing and overriding core methods.
- All computation uses NumPy only for maximal transparency and debuggability.

---

## Example

More examples can be found in examples folder.

```python
import numpy as np
import nn_lib as nn

# Inputs: (2 features, 4 samples)
X = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])

# Targets: (1 output, 4 samples)
Y = np.array([[0, 1, 1, 0]])

# Define the Model
model = nn.models.Sequential([
    nn.layers.Dense(4),
    nn.activations.Sigmoid(), 
    nn.layers.Dense(2),
    nn.activations.Sigmoid(), 
    nn.layers.Dense(1)
], 
loss="SigmoidBCE", 
optimizer='Adam', 
optimizer_params={'learning_rate': 0.001}
)

# Train
print("Training started...")
model.train(X, Y, epochs=5000)
```

See docstrings in each class for detailed parameter and method descriptions.

---

*For code-based exploration and more details, see the class and method docstrings directly in the source code!*
