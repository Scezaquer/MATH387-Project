import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from sgd_model import SGD_MLP
from line_search_mlp import Line_search_MLP, Stochastic_Line_search_MLP
from sgd_line_search_hybrid import SGD_line_search_hybrid_MLP
from pseudo_newton import pseudo_newton_MLP
from sgd_newton_hybrid import SGD_newton_hybrid_MLP
import matplotlib.pyplot as plt
import json


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


def sgd():
    mlp = SGD_MLP(4, (785, 128, 128, 10))  # Don't forget intercept
    acc = mlp.fit(x=X_train, y=y_train, lr=0.01, epochs=1, minibatch_size=64,
                  print_updates=1)

    y_hat = [mlp.predict(x, test=True) for x in X_train]
    train_acc = compute_accuracy(y_train, y_hat)

    y_hat = [mlp.predict(x, test=True) for x in X_test]
    test_acc = compute_accuracy(y_test, y_hat)
    print(f"Train acc: {train_acc}\nTest acc: {test_acc}")
    return acc, train_acc, test_acc


def line_search():
    line_search_mlp = Line_search_MLP(4, (785, 128, 128, 10))
    # pick a subset of 1000 to speed up tests
    """subset_ids = np.random.choice(60000, 1000, replace=False)

    line_search_mlp.fit(x=X_train[subset_ids], y=y_train[subset_ids],
                        minibatch_size=1000, epochs=10, print_updates=1, d=1,
                        n=10)"""
    acc, acc_improvement = line_search_mlp.fit(x=X_train, y=y_train, epochs=10,
                                               print_updates=1, d=1, n=10)

    y_hat = [line_search_mlp.predict(x, test=True) for x in X_train]
    train_acc = compute_accuracy(y_train, y_hat)

    y_hat = [line_search_mlp.predict(x, test=True) for x in X_test]
    test_acc = compute_accuracy(y_test, y_hat)
    print(f"Train acc: {train_acc}\nTest acc: {test_acc}")
    return acc, train_acc, test_acc, acc_improvement


def stochastic_line_search():
    line_search_mlp = Stochastic_Line_search_MLP(4, (785, 128, 128, 10))

    acc, acc_improvement = line_search_mlp.fit(x=X_train, y=y_train,
                        minibatch_size=64, epochs=1, print_updates=1, d=1,
                        n=10)

    y_hat = [line_search_mlp.predict(x, test=True) for x in X_train]
    train_acc = compute_accuracy(y_train, y_hat)

    y_hat = [line_search_mlp.predict(x, test=True) for x in X_test]
    test_acc = compute_accuracy(y_test, y_hat)
    print(f"Train acc: {train_acc}\nTest acc: {test_acc}")
    return acc, train_acc, test_acc, acc_improvement


def hybrid_line_search():
    hybrid = SGD_line_search_hybrid_MLP(4, (785, 128, 128, 10))
    hybrid.fit(x=X_train, y=y_train, x_test=X_test, y_test=y_test, lr=0.01,
               minibatch_size=64, epochs=10, print_updates=1, d=10, n=15,
               fib_epochs=2)

    y_hat = [hybrid.predict(x, test=True) for x in X_train]
    train_acc = compute_accuracy(y_train, y_hat)

    y_hat = [hybrid.predict(x, test=True) for x in X_test]
    test_acc = compute_accuracy(y_test, y_hat)
    print(f"Train acc: {train_acc}\nTest acc: {test_acc}")


def quickprop():
    mlp = pseudo_newton_MLP(4, (785, 128, 128, 10))  # Don't forget intercept
    acc = mlp.fit(x=X_train, y=y_train, lr=0.01, epochs=1, minibatch_size=64,
                  print_updates=1, mu=2)

    y_hat = [mlp.predict(x, test=True) for x in X_train]
    train_acc = compute_accuracy(y_train, y_hat)

    y_hat = [mlp.predict(x, test=True) for x in X_test]
    test_acc = compute_accuracy(y_test, y_hat)
    print(f"Train acc: {train_acc}\nTest acc: {test_acc}")
    return acc, train_acc, test_acc


def sgd_quickprop_hybrid():
    mlp = SGD_newton_hybrid_MLP(4, (785, 128, 128, 10))  # Don't forget intercept
    acc = mlp.fit(x=X_train, y=y_train, lr=0.01, epochs=1, minibatch_size=64,
                  print_updates=1, mu=2, switching_point=35)

    y_hat = [mlp.predict(x, test=True) for x in X_train]
    train_acc = compute_accuracy(y_train, y_hat)

    y_hat = [mlp.predict(x, test=True) for x in X_test]
    test_acc = compute_accuracy(y_test, y_hat)
    print(f"Train acc: {train_acc}\nTest acc: {test_acc}")
    return acc, train_acc, test_acc


all_acc = []
all_train_acc = []
all_test_acc = []
all_acc_improvement = []
# quickprop()
for x in range(5):
    acc, train_acc, test_acc = sgd_quickprop_hybrid()
    all_acc.append(acc)
    all_train_acc.append(train_acc)
    all_test_acc.append(test_acc)
# stochastic_line_search()
# hybrid_line_search()

# Calculate average, highest, and lowest accuracies
average_accuracy = np.mean(all_acc, axis=0)
highest_accuracy = np.max(all_acc, axis=0)
lowest_accuracy = np.min(all_acc, axis=0)

# Plot average accuracy
plt.plot(average_accuracy, label='Average Accuracy')

# Fill between highest and lowest accuracy
plt.fill_between(range(len(average_accuracy)), lowest_accuracy,
                 highest_accuracy, alpha=0.3, label='Variability')

plt.xlabel('Time')
plt.ylabel('Accuracy')
plt.title('Average Training Accuracy with Variability')
plt.legend()
plt.grid(True)
plt.show()

with open("sgd_quickprop_hybrid.json", "w") as f:
    f.write(json.dumps(all_acc))

print(f"Average train acc: {np.mean(all_train_acc)}, "
      f"sigma: {np.std(all_train_acc)}")
print(f"Average test acc: {np.mean(all_test_acc)}, "
      f"sigma: {np.std(all_test_acc)}")
