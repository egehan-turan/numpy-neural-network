import numpy as np
import nn_lib as nn

import matplotlib.pyplot as plt

def create_spiral_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for j in range(classes):
        ix = range(samples * j, samples * (j + 1))
        r = np.linspace(0.05, 3.05, samples)
        # Calculate the starting angle for this class (evenly spaced around 2*pi)
        start_angle = j * (2 * np.pi / classes)
        twist = 6 * np.pi

        t = np.linspace(start_angle, start_angle + twist, samples) + np.random.randn(samples) * 0.2        
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X.T, y

# 1. Generate the data
# Let's create 3 classes with 200 points each
num_classes = 3
X, Y = create_spiral_data(samples=1000, classes=num_classes)

# Define the Model
model = nn.models.Sequential([
    nn.layers.Dense(128), 
    nn.activations.ReLU(),
    nn.layers.Dense(64), 
    nn.activations.ReLU(),
    nn.layers.Dense(3)  
], 
loss="SoftmaxCCE", 
optimizer='Adam', 
optimizer_params={'learning_rate': 0.01} # Slightly higher LR can help speed up initial learning
)

Y_one_hot = np.eye(num_classes)[Y].T

# Train
print("Training started...")
model.train(X, Y_one_hot, epochs=5000)

# Test the results
# --- 1. Define the grid boundaries ---
x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
h = 0.02  # Step size in the mesh

# --- 2. Create the meshgrid ---
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# --- 3. Prepare the grid for the model ---
# Flatten the grid and stack into (2, N) format to match your X shape
grid_points = np.c_[xx.ravel(), yy.ravel()].T

# --- 4. Get Model Predictions ---
# Forward pass through the model
Z_raw = model.forward(grid_points)
# Get the predicted class (index of max value)
Z = np.argmax(Z_raw, axis=0) 

# Reshape back to the grid shape for plotting
Z = Z.reshape(xx.shape)

# --- 5. Plotting ---
plt.figure(figsize=(8, 8))

# Draw the filled contours (the decision regions)
plt.contourf(xx, yy, Z, cmap='jet', alpha=0.3)

# Overlay the original scatter points
plt.scatter(X[0, :], X[1, :], c=Y, s=30, cmap='jet', edgecolors='k')

plt.title("Model Decision Boundaries")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

# Test the results
predictions = model.predict(X)
print("\nFinal Predictions:")
print(predictions)
print("\nAnswer:")
print(Y)