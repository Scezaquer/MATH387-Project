import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sgd_model import SGD_MLP


def load_preprocess_data():
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess and normalize the data
    X_train = X_train.reshape((X_train.shape[0], -1)).astype('float32') / 255.0
    X_test = X_test.reshape((X_test.shape[0], -1)).astype('float32') / 255.0

    # One-hot encode the labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return (X_train, y_train), (X_test, y_test)


def compute_accuracy(y_test, y_hat):
    y_hat_labels = np.argmax(y_hat, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_hat_labels == y_test_labels)

    return accuracy


# Load and preprocess data
(X_train, y_train), (X_test, y_test) = load_preprocess_data()

# Print shapes of train and test data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

mlp = SGD_MLP(4, (785, 128, 128, 10))  # Don't forget intercept
mlp.fit(x=X_train, y=y_train, lr=0.01, epochs=10, minibatch_size=64,
        print_updates=1)

y_hat = [mlp.predict(x, test=True) for x in X_train]
train_acc = compute_accuracy(y_train, y_hat)

y_hat = [mlp.predict(x, test=True) for x in X_test]
test_acc = compute_accuracy(y_test, y_hat)
print(f"Train acc: {train_acc}\nTest acc: {test_acc}")
