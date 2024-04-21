import numpy as np
from tqdm import tqdm


def compute_accuracy(y_test, y_hat):
    y_hat_labels = np.argmax(y_hat, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_hat_labels == y_test_labels)

    return accuracy


def compute_loss(y_test, y_hat):
    return sum([y@label for y, label in zip(y_hat, 2*y_test-1)])


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def sigmoid_derivative(x):
    """Numerically stable derivative of the sigmoid function."""
    a = sigmoid(x)
    return a * (1 - a)


class Stochastic_Line_search_MLP:
    def __init__(self, nbr_layers, units_per_layer):
        assert nbr_layers == len(units_per_layer)
        self.activations = []
        self.layers = []
        for i in range(nbr_layers-1):
            layer = np.random.rand(units_per_layer[i+1], units_per_layer[i])
            layer = (layer - 0.5)*0.2
            self.layers.append(layer)

    def fit(self, x, y, epochs, minibatch_size=1000,
            print_updates=1, d=10, n=10):
        # Labels must be one-hot encoded
        acc = []
        acc_improvement = []
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

                # fibonacci search
                # we assume the min is in the direction of the gradient and
                # within some arbitrary distance d
                cumulative_grad = [i/minibatch_size for i in cumulative_grad]
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

                pred_x1 = [self.predict(data, using_weights=x1)
                           for data in minibatch_x]
                pred_x2 = [self.predict(data, using_weights=x2)
                           for data in minibatch_x]

                fx1 = -compute_loss(minibatch_y, pred_x1)
                fx2 = -compute_loss(minibatch_y, pred_x2)
                print(f"iter {1}, acc1: {-fx1} acc2: {-fx2}")

                for i in range(2, n-1):
                    if fx2 > fx1:
                        best_pt = x1
                        b = x2
                        fx2 = fx1

                        x1 = [fib_numbers[n-i-1]/fib_numbers[n-i+1]*(b[j]-a[j])
                              + a[j] for j, _ in enumerate(self.layers)]
                        pred_x1 = [self.predict(data, using_weights=x1)
                                   for data in minibatch_x]
                        fx1 = -compute_loss(minibatch_y, pred_x1)

                    else:
                        best_pt = x2
                        a = x1
                        fx1 = fx2

                        x2 = [fib_numbers[n-i-1]/fib_numbers[n-i+1]*(b[j]-a[j])
                              + a[j] for j, _ in enumerate(self.layers)]
                        pred_x2 = [self.predict(data, using_weights=x2)
                                   for data in minibatch_x]
                        fx2 = -compute_loss(minibatch_y, pred_x2)
                    print(f"iter {i}, acc1: {-fx1} acc2: {-fx2}")
                    accuracy = compute_accuracy(minibatch_y, pred_x2)
                    if i == 2:
                        acc_improvement.append(accuracy)
                    if i == n-2:
                        acc_improvement[-1] = accuracy - acc_improvement[-1]
                    print(f"acc: {accuracy}")

                # pick the best of the two endpoints
                self.layers = best_pt
                acc.append(accuracy)

            if print_updates and not epoch % print_updates:
                print(f"[TRAINING] epoch = {epoch}, loss={cumulative_loss}")
        return acc, acc_improvement

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


class Line_search_MLP:
    def __init__(self, nbr_layers, units_per_layer):
        assert nbr_layers == len(units_per_layer)
        self.activations = []
        self.layers = []
        for i in range(nbr_layers-1):
            layer = np.random.rand(units_per_layer[i+1], units_per_layer[i])
            layer = (layer - 0.5)*0.2
            self.layers.append(layer)

    def fit(self, x, y, epochs, print_updates=1, d=10, n=10):
        # Labels must be one-hot encoded
        acc = []
        acc_improvement = []
        for epoch in range(epochs):
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

            pred_x1 = [self.predict(data, using_weights=x1) for data in x]
            pred_x2 = [self.predict(data, using_weights=x2) for data in x]

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
                    pred_x1 = [self.predict(data, using_weights=x1)
                               for data in x]
                    fx1 = -compute_loss(y, pred_x1)

                else:
                    best_pt = x2
                    a = x1
                    fx1 = fx2

                    x2 = [fib_numbers[n-i-1]/fib_numbers[n-i+1]*(b[j]-a[j]) +
                          a[j] for j, _ in enumerate(self.layers)]
                    pred_x2 = [self.predict(data, using_weights=x2)
                               for data in x]
                    fx2 = -compute_loss(y, pred_x2)
                print(f"iter {i}, acc1: {-fx1} acc2: {-fx2}")
                accuracy = compute_accuracy(y, pred_x2)
                if i == 2:
                    acc_improvement.append(accuracy)
                if i == n-2:
                    acc_improvement[-1] = accuracy - acc_improvement[-1]
                print(f"acc: {accuracy}")

            acc.append(accuracy)
            # pick the best of the two endpoints
            self.layers = best_pt

            if print_updates and not epoch % print_updates:
                print(f"[TRAINING] epoch = {epoch}, loss={cumulative_loss}")
        return acc, acc_improvement

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


if __name__ == "__main__":
    # construct the XOR dataset to test the implementation on a simple case
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    mlp = Line_search_MLP(4, (3, 3, 3, 2))
    mlp.fit(X, y, 0.5, 50000, 4, 1000)

    print(f"Input: [0, 0], Predicted: {mlp.predict([0, 0])} Correct: [1, 0]")
    print(f"Input: [0, 1], Predicted: {mlp.predict([0, 1])} Correct: [0, 1]")
    print(f"Input: [1, 0], Predicted: {mlp.predict([1, 0])} Correct: [0, 1]")
    print(f"Input: [1, 1], Predicted: {mlp.predict([1, 1])} Correct: [1, 0]")
