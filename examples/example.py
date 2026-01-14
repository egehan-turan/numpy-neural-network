import numpy as np
import nn_lib as nn

# XOR Data
# Inputs: (2 features, 4 samples)
X = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])

# Targets: (1 output, 4 samples)
Y = np.array([[0, 1, 1, 0]])

# Define the Model
model = nn.models.Sequential([
    nn.layers.Dense(output_dim=4, input_dim=2), # Hidden Layer
    nn.activations.ReLU(),
    nn.layers.Dense(output_dim=1),              # Output Layer
    nn.activations.Sigmoid()
], 
loss="MSE", 
optimizer='SGD', 
#optimizer_params={'learning_rate': 0.1}
)

# Train
print("Training started...")
model.train(X, Y, epochs=3000)

# Test the results
predictions = model.forward(X)
print("\nFinal Predictions:")
print(predictions)
print("\nRounded Predictions:")
print(np.round(predictions))