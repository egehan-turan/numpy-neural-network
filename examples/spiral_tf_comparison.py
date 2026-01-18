from gc import callbacks
import numpy as np
import nn_lib as nn
import tensorflow as tf

import matplotlib.pyplot as plt

NUM_CLASSES = 3

def main():
    # Generate the data
    X, Y = create_spiral_data(samples=1000, classes=NUM_CLASSES)

    # Define the Model
    nn_model = nn.models.Sequential([
        nn.layers.Dense(64, activation='ReLU'), 
        nn.layers.Dense(32, activation='ReLU'), 
        nn.layers.Dense(3)  
    ], 
    loss="SoftmaxCCE", 
    optimizer='Adam', 
    optimizer_params={'learning_rate': 0.001},
    sample_axis=0
    )

    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu'), 
        tf.keras.layers.Dense(32, activation='relu'), 
        tf.keras.layers.Dense(3),
        tf.keras.layers.Softmax()
    ], )

    # Train neural network
    tf_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy"
    )

    Y_one_hot = np.eye(NUM_CLASSES)[Y]
    
    # Train
    res = nn_model.train(X, Y_one_hot, epochs=1000, loss_threshold=0.001)

    start_time = tf.timestamp()
    tf_model.fit(X, Y_one_hot, epochs=1000, callbacks=[StopAtLossThreshold(0.001)])
    end_time = tf.timestamp()

    plot_data_comparison(X.T, Y, nn_model, tf_model, res['training_time'], end_time - start_time)


class StopAtLossThreshold(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(StopAtLossThreshold, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        # Retrieve the current loss from the logs dictionary
        current_loss = logs.get('loss')
        
        if current_loss is not None:
            if current_loss <= self.threshold:
                print(f"\nLoss reached {current_loss:.4f}, which is below threshold {self.threshold}. Stopping training.")
                self.model.stop_training = True


def plot_data_comparison(X, Y, nn_model, tf_model, nn_training_time, tf_training_time):
    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5))

    # Call the modified function twice
    plot_data(X, Y, nn_model, ax=ax1, sample_axis=1)
    ax1.set_title(f"NN Model Training Time: {nn_training_time:.2f}s")

    plot_data(X, Y, tf_model, ax=ax2, sample_axis=0)
    ax2.set_title(f"TF Model Training Time: {tf_training_time:.2f}s")

    plt.tight_layout()
    plt.show()


def plot_data(X, Y, model, ax=None, sample_axis=1):
    # If no axis is provided (standalone call), create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Define the grid boundaries (using your specific X indexing)
    x_min, x_max = X[0, :].min() - 0.5, X[0, :].max() + 0.5
    y_min, y_max = X[1, :].min() - 0.5, X[1, :].max() + 0.5
    h = 0.02

    # Create the meshgrid 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # Forward pass and Argmax (using your axis=0 logic)
    Z_raw = model.predict(grid_points)
    Z = np.argmax(Z_raw, axis=1) 
        
    Z = Z.reshape(xx.shape)

    # Plot onto the specific axis
    ax.contourf(xx, yy, Z, cmap='jet', alpha=0.3)
    ax.scatter(X[0, :], X[1, :], c=Y, s=30, cmap='jet', edgecolors='k')

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())


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
    return X, y


if __name__ == '__main__':
    main()