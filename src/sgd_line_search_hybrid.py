import numpy as np
from tqdm import tqdm


def compute_loss(y_test, y_hat):
    return sum([y@label for y, label in zip(y_hat, 2*y_test-1)])


def compute_accuracy(y_test, y_hat):
    y_hat_labels = np.argmax(y_hat, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_hat_labels == y_test_labels)

    return accuracy


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivative(x):
    """Numerically stable derivative of the sigmoid function."""
    a = sigmoid(x)
    return a * (1 - a)


class SGD_line_search_hybrid_MLP:
    def __init__(self, nbr_layers, units_per_layer):
        assert nbr_layers == len(units_per_layer)
        self.activations = []
        self.layers = []
        for i in range(nbr_layers-1):
            layer = np.random.rand(units_per_layer[i+1], units_per_layer[i])
            layer = layer - 0.5
            self.layers.append(layer)

    def fit(self, x, y, x_test, y_test, lr, epochs, minibatch_size,
            print_updates=1000, d=1, n=10, fib_epochs=2):
        # Labels must be one-hot encoded
        for epoch in range(epochs):
            cumulative_loss = 0
            for minibatch in tqdm(range(int(len(x)/minibatch_size))):
                cumulative_grad = [np.zeros(i.shape) for i in self.layers]
                minibatch_ids = np.random.choice(len(x), minibatch_size, False)
                minibatch_x = x[minibatch_ids]
                minibatch_y = y[minibatch_ids]

                for j in range(minibatch_size):
                    # Forward pass
                    # Note: the activations are stored in self.activations
                    y_hat = self.predict(minibatch_x[j])

                    # Compute loss
                    loss = -np.log(y_hat@minibatch_y[j]) / len(minibatch_y[j])
                    cumulative_loss += loss

                    # Backward pass
                    # Gradients at the last layer
                    error = y_hat - minibatch_y[j]
                    gradients = [
                        np.atleast_2d(error*sigmoid_derivative(y_hat))]

                    # Gradients at hidden layers
                    for layer, activation in zip(self.layers[::-1][:-1],
                                                 self.activations[::-1][1:]):
                        grad = gradients[-1]@layer * \
                            sigmoid_derivative(activation)
                        gradients.append(grad)
                    gradients = gradients[::-1]

                    # Update the sum of gradients
                    cumulative_grad = \
                        [x + y.T@z for x, y, z in
                         zip(cumulative_grad, gradients, self.activations)]

                # Update the weights from the total of the gradients
                for i, layer in enumerate(self.layers):
                    self.layers[i] = layer - lr*cumulative_grad[i]

            if print_updates and not epoch % print_updates:
                print(f"[TRAINING] epoch = {epoch}, loss={cumulative_loss}")

        pred = [self.predict(data, test=True) for data in x]
        print(f"Train accuracy before fib: {compute_accuracy(y, pred)}")
        pred = [self.predict(data, test=True) for data in x_test]
        print(f"Test accuracy before fib: {compute_accuracy(y_test, pred)}")

        for epoch in range(fib_epochs):
            cumulative_loss = 0
            cumulative_grad = [np.zeros(i.shape) for i in self.layers]
            for image, label in tqdm(zip(x, y), total=len(y)):
                # Forward pass
                # Note: the activations are stored in self.activations
                y_hat = self.predict(image)

                # Compute loss
                loss = -np.log(y_hat@label) / len(label)
                cumulative_loss += loss

                # Backward pass
                # Gradients at the last layer
                error = y_hat - label
                gradients = [
                    np.atleast_2d(error*sigmoid_derivative(y_hat))]

                # Gradients at hidden layers
                for layer, activation in zip(self.layers[::-1][:-1],
                                             self.activations[::-1][1:]):
                    grad = gradients[-1]@layer * \
                        sigmoid_derivative(activation)
                    gradients.append(grad)
                gradients = gradients[::-1]

                # Update the sum of gradients
                cumulative_grad = \
                    [x + y.T@z for x, y, z in
                        zip(cumulative_grad, gradients, self.activations)]

            # fibonacci search
            # we assume the min is in the direction of the gradient and within
            # some arbitrary distance d
            cumulative_grad = [i/len(y) for i in cumulative_grad]
            a = self.layers
            b = [layer - d*cumulative_grad[i]
                 for i, layer in enumerate(self.layers)]

            fib_numbers = [1, 1]
            for _ in range(n-1):
                fib_numbers += [fib_numbers[-1] + fib_numbers[-2]]

            x1 = [fib_numbers[n-2]/fib_numbers[n]*(b[j]-a[j]) + a[j]
                  for j, _ in enumerate(self.layers)]
            x2 = [fib_numbers[n-1]/fib_numbers[n]*(b[j]-a[j]) + a[j]
                  for j, _ in enumerate(self.layers)]

            pred_x1 = [self.predict(data, using_weights=x1, test=True)
                       for data in x]
            pred_x2 = [self.predict(data, using_weights=x2, test=True)
                       for data in x]

            fx1 = -compute_loss(y, pred_x1)
            fx2 = -compute_loss(y, pred_x2)
            print(f"iter {1}, acc1: {-fx1} acc2: {-fx2}")

            for i in range(2, n-1):
                if fx2 > fx1:
                    best_pt = x1
                    b = x2
                    fx2 = fx1

                    x1 = [fib_numbers[n-i-1]/fib_numbers[n-i+1]*(b[j]-a[j]) +
                          a[j] for j, _ in enumerate(self.layers)]
                    pred_x1 = [self.predict(data, using_weights=x1, test=True)
                               for data in x]
                    fx1 = -compute_loss(y, pred_x1)

                else:
                    best_pt = x2
                    a = x1
                    fx1 = fx2

                    x2 = [fib_numbers[n-i-1]/fib_numbers[n-i+1]*(b[j]-a[j]) +
                          a[j] for j, _ in enumerate(self.layers)]
                    pred_x2 = [self.predict(data, using_weights=x2, test=True)
                               for data in x]
                    fx2 = -compute_loss(y, pred_x2)
                print(f"iter {i}, acc1: {-fx1} acc2: {-fx2}")
                print(f"acc: {compute_accuracy(y, pred_x2)}")

            # pick the best of the two endpoints
            self.layers = best_pt

            if print_updates and not epoch % print_updates:
                print(f"[TRAINING] epoch = {epoch}, loss={cumulative_loss}")

    def predict(self, x, test=False, using_weights=None):
        x = np.append(x, 1)  # Intercept
        if not test:
            self.activations = [np.atleast_2d(x)]
        layers = self.layers if using_weights is None else using_weights
        for i in layers:
            x = sigmoid(i@x)
            if not test:
                self.activations.append(np.atleast_2d(x))
        return x
