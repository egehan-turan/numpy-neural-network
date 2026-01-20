import tensorflow as tf
from tensorflow import keras

def main():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = (x_train.astype('float32') / 255.0)
    x_test = (x_test.astype('float32') / 255.0)

    # Convert labels to one-hot vectors
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Build the model using Sequential
    model = tf.keras.models.load_model('cifar10_model.keras')

    print("Evaluating Test Set:")
    _, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")

    print("\nEvaluating Training Set:")
    _, train_acc = model.evaluate(x_train, y_train, verbose=0)
    print(f"Final Train Accuracy: {train_acc * 100:.2f}%")

if __name__ == '__main__':
    main()