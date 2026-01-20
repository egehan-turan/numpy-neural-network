import tensorflow as tf
from tensorflow import keras
import numpy as np
from tqdm import tqdm

import nn_lib as nn

def main():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = (x_train.astype('float32') / 255.0).transpose((3,1,2,0))
    x_test = (x_test.astype('float32') / 255.0).transpose((3,1,2,0))

    # Convert labels to one-hot vectors
    y_train = keras.utils.to_categorical(y_train, 10).T
    y_test = keras.utils.to_categorical(y_test, 10).T

    # Build the model using Sequential
    model = nn.models.Model.load('cifar10_my_model.pkl')

    print("Evaluating Test Set:")
    accuracy_test = evaluate_in_batches(model, x_test, y_test)
    print(f"Final Test Accuracy: {accuracy_test * 100:.2f}%")

    print("\nEvaluating Training Set:")
    accuracy_train = evaluate_in_batches(model, x_train, y_train)
    print(f"Final Train Accuracy: {accuracy_train * 100:.2f}%")


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

if __name__ == '__main__':
    main()