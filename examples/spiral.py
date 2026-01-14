import numpy as np
import matplotlib.pyplot as plt

def create_spiral_data(samples, classes):
    X = np.zeros((samples * classes, 2))
    y = np.zeros(samples * classes, dtype='uint8')
    for j in range(classes):
        ix = range(samples * j, samples * (j + 1))
        r = np.linspace(0.0, 1, samples)
        # Calculate the starting angle for this class (evenly spaced around 2*pi)
        start_angle = j * (2 * np.pi / classes)
        twist = np.pi

        t = np.linspace(start_angle, start_angle + twist, samples) + np.random.randn(samples) * 0.2        
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X.T, y

# 1. Generate the data
# Let's create 3 classes with 200 points each
X, y = create_spiral_data(samples=200, classes=3)

# 2. Create the plot
plt.figure(figsize=(8, 8))

# X[0, :] are the x-coordinates
# X[1, :] are the y-coordinates
# 'c=y' colors the points by their class label
plt.scatter(X[0, :], X[1, :], c=y, s=20, cmap='jet', edgecolors='k', alpha=0.7)

# 3. Add styling
plt.title("Synthetic Spiral Dataset", fontsize=14)
plt.xlabel("Feature 1 ($x$)", fontsize=12)
plt.ylabel("Feature 2 ($y$)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Show the plot
plt.show()