"""
XOR test example
"""
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

# Test the results
predictions = model.predict(X)
print("\nFinal Predictions:")
print(predictions)
print("\nRounded Predictions:")
print(np.round(predictions))
print("\nAnswer:")
print(Y)