import numpy as np
from tqdm import tqdm


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


class pseudo_newton_MLP:  # Quickprop
    def __init__(self, nbr_layers, units_per_layer):
        assert nbr_layers == len(units_per_layer)
        self.activations = []
        self.layers = []
        for i in range(nbr_layers-1):
            layer = np.random.rand(units_per_layer[i+1], units_per_layer[i])
            layer = layer - 0.5
            self.layers.append(layer)

    def fit(self, x, y, lr, epochs, minibatch_size, print_updates=1000, mu=2):
        # Labels must be one-hot encoded
        for epoch in range(epochs):
            cumulative_loss = 0

            delta_w_prev = [np.zeros(i.shape) for i in self.layers]
            g_prev = [np.zeros(i.shape) for i in self.layers]

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
                delta_w = [np.zeros(i.shape) for i in self.layers]
                for i, layer in enumerate(self.layers):
                    delta_w[i] = cumulative_grad[i] / (g_prev[i] - cumulative_grad[i]) * delta_w_prev[i]

                    overflow_mask = np.isnan(delta_w[i]) | (abs(delta_w[i]) > abs(delta_w_prev[i]))
                    delta_w[i][overflow_mask] = mu*delta_w_prev[i][overflow_mask]

                    # GD step for weights that didn't change
                    zero_mask = (delta_w[i] == 0).astype(int)
                    delta_w[i] = delta_w[i] - lr*cumulative_grad[i]*zero_mask

                    self.layers[i] = layer + delta_w[i]

                delta_w_prev = delta_w
                g_prev = cumulative_grad

                pred = [self.predict(data, test=True) for data in minibatch_x]
                print("Train acc: "
                      f"{compute_accuracy(minibatch_y, pred)}")

            if print_updates and not epoch % print_updates:
                print(f"[TRAINING] epoch = {epoch}, loss={cumulative_loss}")

    def predict(self, x, test=False):
        x = np.append(x, 1)  # Intercept
        if not test:
            self.activations = [np.atleast_2d(x)]
        for i in self.layers:
            x = sigmoid(i@x)
            if not test:
                self.activations.append(np.atleast_2d(x))
        return x


if __name__ == "__main__":
    # construct the XOR dataset to test the implementation on a simple case
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

    mlp = pseudo_newton_MLP(4, (3, 3, 3, 2))
    mlp.fit(X, y, 0.5, 50000, 4, 1000)

    print(f"Input: [0, 0], Predicted: {mlp.predict([0, 0])} Correct: [1, 0]")
    print(f"Input: [0, 1], Predicted: {mlp.predict([0, 1])} Correct: [0, 1]")
    print(f"Input: [1, 0], Predicted: {mlp.predict([1, 0])} Correct: [0, 1]")
    print(f"Input: [1, 1], Predicted: {mlp.predict([1, 1])} Correct: [1, 0]")
