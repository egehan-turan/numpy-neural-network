import numpy as np
import nn_lib as nn

import matplotlib.pyplot as plt

NUM_CLASSES = 3

def main():
    # Generate the data
    X, Y = create_spiral_data(samples=1000, classes=NUM_CLASSES)

    # Define the Model
    model = nn.models.Sequential([
        nn.layers.Dense(64, activation='ReLU'), 
        nn.layers.Dense(32, activation='ReLU'), 
        nn.layers.Dense(3)  
    ], 
    loss="SoftmaxCCE", 
    optimizer='Adam', 
    optimizer_params={'learning_rate': 0.001} 
    )

    Y_one_hot = np.eye(NUM_CLASSES)[Y].T

    # Train
    model.train(X, Y_one_hot, epochs=1000)

    plot_spiral_data(X, Y, model)

    # Save model
    # model.save('spiral_model.pkl')


def plot_spiral_data(X, Y, model):

    # Define the grid boundaries 
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.02  # Step size in the mesh

    # Create the meshgrid 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # Flatten the grid and stack into (2, N) format to match your X shape
    grid_points = np.c_[xx.ravel(), yy.ravel()].T

    # Forward pass through the model
    Z_raw = model.predict(grid_points)
    # Get the predicted class
    Z = np.argmax(Z_raw, axis=0) 

    # Reshape back to the grid shape for plotting
    Z = Z.reshape(xx.shape)

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

    return


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

if __name__ == '__main__':
    main()