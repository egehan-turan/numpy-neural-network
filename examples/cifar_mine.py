import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm

import nn_lib as nn

def evaluate_in_batches(model, x_data, y_data, batch_size=128):
    # Since samples is the last dimension: shape[3]
    num_samples = x_data.shape[3]
    total_correct = 0
    
    pbar = tqdm(range(0, num_samples, batch_size), desc="Evaluating")

    for i in pbar:
        end = min(i + batch_size, num_samples)
        
        # Slice along the last axis (samples)
        # Slicing syntax: [all_channels, all_height, all_width, batch_range]
        batch_x = x_data[:, :, :, i:end]
        batch_y = y_data[:, i:end] 
        
        # Predict on the reshaped batch
        batch_y_hat = model.predict(batch_x)
        
        pred_classes = np.argmax(batch_y_hat, axis=0)
        true_classes = np.argmax(batch_y, axis=0)
        
        total_correct += np.sum(pred_classes == true_classes)
        
        # Live metrics
        current_acc = total_correct / end
        pbar.set_postfix({"Err": f"{(1.0 - current_acc) * 100:.2f}%"})
        
    return total_correct / num_samples

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to [0, 1]
x_train = (x_train.astype('float32') / 255.0).transpose((3,1,2,0))
x_test = (x_test.astype('float32') / 255.0).transpose((3,1,2,0))

# Convert labels to one-hot vectors
y_train = keras.utils.to_categorical(y_train, 10).T
y_test = keras.utils.to_categorical(y_test, 10).T


# Build the model using Sequential
model = nn.models.Sequential([
    # Block 1
    nn.layers.FastConv2D(32, (3,3), use_bias=False),
    nn.layers.BatchNorm(),
    nn.activations.ReLU(),
    nn.layers.FastConv2D(32, (3,3), use_bias=False),
    nn.layers.BatchNorm(),
    nn.activations.ReLU(),
    nn.layers.FastMaxPooling2D((2,2)),
    nn.layers.Dropout(0.2),
    
    # Block 2
    nn.layers.FastConv2D(64, (3,3), use_bias=False),
    nn.layers.BatchNorm(),
    nn.activations.ReLU(),
    nn.layers.FastConv2D(64, (3,3), use_bias=False),
    nn.layers.BatchNorm(),
    nn.activations.ReLU(),
    nn.layers.FastMaxPooling2D((2,2)),
    nn.layers.Dropout(0.3),
    
    # Block 3
    nn.layers.FastConv2D(128, (3,3), use_bias=False),
    nn.layers.BatchNorm(),
    nn.activations.ReLU(),
    nn.layers.FastConv2D(128, (3,3), use_bias=False),
    nn.layers.BatchNorm(),
    nn.activations.ReLU(),
    nn.layers.FastMaxPooling2D((2,2)),
    nn.layers.Dropout(0.4),
    
    # Dense layers
    nn.layers.Flatten(),
    nn.layers.Dense(128, use_bias=False),
    nn.layers.BatchNorm(),
    nn.activations.ReLU(),
    nn.layers.Dropout(0.5),
    nn.layers.Dense(10)
],
optimizer = 'Adam',
loss = 'SoftmaxCCE', 
optimizer_params={'learning_rate': 0.001}
)

# Train the model
history = model.train(
    x_train, y_train,
    batch_size=64,
    epochs=100
)

model.save('cifar10_my_model.pkl')

print("Evaluating Test Set:")
accuracy_test = evaluate_in_batches(model, x_test, y_test)
print(f"Final Test Accuracy: {accuracy_test * 100:.2f}%")

print("Evaluating Training Set:")
accuracy_test = evaluate_in_batches(model, x_train, y_train)
print(f"Final Training Accuracy: {accuracy_test * 100:.2f}%")

'''
# Evaluate on test set
y_hat = model.predict(x_test)
test_loss = nn.losses.CCE().loss(y_test, y_hat)
print(f"Test loss: {test_loss:.4f}")

predicted_classes = np.argmax(y_hat, axis=1)
true_classes = np.argmax(y_test, axis=1)

# 2. Compare and calculate the mean
accuracy = np.mean(predicted_classes == true_classes)

print(f"Test Accuracy: {accuracy * 100:.22f}%")
'''