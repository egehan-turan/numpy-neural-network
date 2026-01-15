"""
Train on MNIST dataset and save the model
"""
import nn_lib as nn
import tensorflow as tf

# Use MNIST handwriting dataset
mnist = tf.keras.datasets.mnist

# Prepare data for training
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)

x_train = x_train.transpose(3, 1, 2, 0)
y_train = y_train.transpose(1, 0)

# Create a CNN
model = nn.models.Sequential([
    nn.layers.Conv2D(32, (3, 3)),
    nn.activations.ReLU(),
    nn.layers.MaxPooling2D(pool_size=(2, 2)),
    nn.layers.Flatten(),
    nn.layers.Dense(128),
    nn.activations.ReLU(),
    nn.layers.Dense(10)
], 
loss="SoftmaxCCE", 
optimizer='Adam', 
optimizer_params={'learning_rate': 0.001}
)

print("Training started...")
model.train(x_train, y_train, epochs=10)

# Save model
model.save('mnist_model.pkl')