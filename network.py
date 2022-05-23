import numpy as np
import time
from sklearn.utils import shuffle
import wandb


class Layer:
    """
    Base Layer class for all layers.
    """

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        """
        Forward pass of the layer.

        We expect it to return the output of the layer.
        """
        pass

    def backward(self, output_gradient, learning_rate):
        """
        Backward pass of the layer.

        We expect it to return the input gradient of the layer (to be used by the layers preceding it).
        """
        pass


class Dense(Layer):
    """
    Implementation of a Layer that uses weights and biases to compute the output.
    """

    def __init__(self, input_size, output_size, use_bias=True, momentum=0.0):
        # Weights are initialized using a random normal distribution.
        # randn() returns a matrix with size (output_size, input_size)
        self.weights = np.random.randn(output_size, input_size)
        # For biases we initialize them to zeros.
        if use_bias:
            self.bias = np.random.randn(output_size, 1)
        else:
            self.bias = np.zeros((output_size, 1))
        self.use_bias = use_bias
        # COMMENT
        self.momentum = momentum
        self.weight_momentums = np.zeros_like(self.weights)
        self.bias_momentums = np.zeros_like(self.bias)

    def forward(self, input):
        self.input = input
        # Multiply the input by the weights and add the bias.
        self.output = np.dot(self.weights, self.input)
        if self.use_bias:
            self.output += self.bias
        return self.output

    def backward(self, output_gradient, learning_rate):
        # When calculating the gradient we multiply the output gradient that was
        # given to us by the derivative of whatever gradient we're computing.

        # Gradient of weights is the dot product of the output gradient and the input.
        # Since y = wx + b, we have dy/dw = x.
        weights_gradient = np.dot(output_gradient, self.input.T)
        # Gradient of the bias is the output gradient.
        # Since y = wx + b, we have dy/db = 1.
        bias_gradient = output_gradient

        # Since y = wx + b, we have dy/dx = w.
        input_gradient = np.dot(self.weights.T, output_gradient)

        # Update the weights and the bias.
        weight_updates = self.momentum * self.weight_momentums - \
            learning_rate * weights_gradient
        self.weight_momentums = weight_updates
        self.weights += weight_updates

        if self.use_bias:
            bias_updates = self.momentum * self.bias_momentums - learning_rate * bias_gradient
            self.bias_momentums = bias_updates
            self.bias += bias_updates

        return input_gradient


class Activation(Layer):
    """
    Activation Layer to be used after a Dense layer.

    You provide it with the activation function and its derivative.
    """

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient, learning_rate):
        # When calculating the gradient we multiply the output gradient that was
        # given to us by the derivative of our activation function.
        # We provide the activation function input, since we want to use its derivative.
        return np.multiply(output_gradient, self.activation_prime(self.input))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


class Sigmoid(Activation):
    """
    Sigmoid Activation Layer.
    """

    def __init__(self):
        super().__init__(sigmoid, sigmoid_prime)


class Softmax(Layer):
    """
    Softmax Activation Layer.

    Should be used for output layer only.
    """

    def forward(self, input):
        self.input = input
        tmp = np.exp(self.input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)


def array_to_string(array):
    return np.array2string(array, formatter={'float_kind': '{0:.3f}'.format}).replace('\n', '').replace(' ', ', ')


class Network:
    """
    Network gathers all the layers and provides helpful methods for training and testing the model.
    """

    def __init__(self, network_dimensions, activation, use_bias=True, use_softmax=False, momentum=0.0):
        """
        `network_dimensions` is a list of the number of neurons in each layer.
        For example, if the network has 2 layers, `network_dimensions` would be [2, 3, 1].

        `activation` is the Activation class to be used between Dense layers.
        """
        self.layers = []
        for i in range(len(network_dimensions) - 1):
            self.layers.append(
                Dense(network_dimensions[i], network_dimensions[i + 1], use_bias=use_bias, momentum=momentum))
            if use_softmax and i == len(network_dimensions) - 2:
                self.layers.append(Softmax())
            else:
                self.layers.append(activation())

    def predict(self, input):
        """
        Runs a forward pass through all the layers of the network.

        Outputs the output of the final layer.
        """
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def confusion_matrix(self, y_pred, y_true, num_classes):
        """
        Returns a confusion matrix based on the given true and predicted labels.
        """
        matrix = np.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                matrix[i, j] = np.sum(np.logical_and(y_pred == i, y_true == j))
        return matrix

    def precisions(self, matrix):
        """
        Returns the precision for each class.
        """
        if np.sum(matrix, axis=0).all() == 0:
            return np.zeros(matrix.shape[0])
        return np.diag(matrix) / np.sum(matrix, axis=0)

    def recalls(self, matrix):
        """
        Returns the recall for each class.
        """
        return np.diag(matrix) / np.sum(matrix, axis=1)

    def fscore(self, precisions, recalls):
        """
        Returns the fscore for each class.
        """
        return 2 * precisions * recalls / (precisions + recalls)

    def train(self, loss, loss_prime, x_train, y_train, is_iris, x_test=None, y_test=None, epochs=1000, learning_rate=0.01, random_order=True, goal_error=None, log_to_file=False):
        """
        Runs a training loop (forward pass and backward pass) on the network.
        Additionally logs the training progress to a file / wandb.
        """
        message = ""
        for e in range(epochs):
            # Training
            error = 0
            if random_order:
                x_train, y_train = shuffle(x_train, y_train)
            for x, y in zip(x_train, y_train):
                # forward pass
                output = self.predict(x)

                # error
                error += loss(y, output)

                # backward pass
                grad = loss_prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)

            # Calculating precision, recall and fscore
            predictions = []
            # We're predicting the prediction for each sample in the test set...
            for x, y in zip(x_test, y_test):
                predictions.append(self.predict(x))

            # and after that we figure out which class was chosen by each prediction.
            predictions_indexes = np.argmax(predictions, axis=1)
            y_test_indexes = np.argmax(y_test, axis=1)

            # Reshaping stuff...
            predictions_indexes = np.reshape(
                predictions_indexes, (len(predictions_indexes)))
            y_test_indexes = np.reshape(y_test_indexes, (len(y_test_indexes)))

            # Calculating the confusion matrix
            matrix = self.confusion_matrix(
                y_test_indexes, predictions_indexes, 3 if is_iris else 4)

            # Calculating the precision, recall and fscore for each class
            precisions = self.precisions(matrix)
            recalls = self.recalls(matrix)
            fscores = self.fscore(precisions, recalls)

            # Taking the average for all classes
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            fscore = np.mean(fscores)

            # Logging stuff to wandb
            wandb.log({'loss': error, 'precision': precision,
                      'recall': recall, 'fscore': fscore})

            # If we've finished training, we send confusion matrix to wandb as well.
            if (e == epochs - 1 or (goal_error and error < goal_error)):
                if is_iris:
                    wandb.log({'conf_mat': wandb.plot.confusion_matrix(probs=None,
                                                                       y_true=y_test_indexes, preds=predictions_indexes,
                                                                       class_names=["setosa", "versicolor", "virginica"])})
                else:
                    wandb.log({'conf_mat': wandb.plot.confusion_matrix(probs=None,
                                                                       y_true=y_test_indexes, preds=predictions_indexes,
                                                                       class_names=["1000", "0100", "0010", "0001"])})

            # Logging stuff to a file
            if e % 50 == 0:
                print("Epoch:", e, "Error:", error)
                print(matrix)
                if log_to_file:
                    message += "Epoch: " + str(e) + \
                        " Error: " + str(error) + "\n"

            # If we've reach goal, we finish eariler.
            if goal_error is not None:
                if error < goal_error:
                    print("Goal error reached!")
                    break

        if log_to_file:
            filename = time.strftime("%Y%m%d-%H%M%S")
            with open("training-" + filename, "w") as f:
                f.write(message)

    def test(self, x_test, y_test):
        message = ""
        error = 0
        for x, y in zip(x_test, y_test):
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            error += mse(y, output)

            message += "Input:       " + array_to_string(x) + "\n"
            message += "Pred output: " + array_to_string(output) + "\n"
            message += "True output: " + array_to_string(y) + "\n"
            message += "Error: " + str(mse(y, output)) + "\n"

        error /= len(x_test)

        message += "\n"
        message += "------------------------------------------------------\n"
        message += "\n"
        message += "Global error: " + str(error) + "\n"

        for i, layer in enumerate(self.layers[::2]):
            message += "Weights on layer " + \
                str(i) + ": " + "\n" + str(layer.weights) + "\n"
            message += "Biases on layer " + \
                str(i) + ": " + "\n" + str(layer.bias) + "\n"

        filename = time.strftime("%Y%m%d-%H%M%S")
        with open("testing-" + filename, "w") as f:
            f.write(message)
