import numpy as np
import pandas as pd
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import wandb
import fire

from network import Network, Sigmoid, mse, mse_prime


def classes_to_one_hot(classes, num_classes):
    classes = classes.to_numpy().reshape(classes.shape[0])
    one_hot = np.zeros((classes.size, num_classes))
    one_hot[np.arange(classes.size), classes] = 1
    return one_hot


class Program(object):
    def train(self, dataset, network_file=None, learning_rate=0.2, momentum=0.0, use_bias=True, epochs=1000, test_size=0.2, random_order=True, goal_error=None, log_to_file=False):
        config = {
            "learning_rate": learning_rate,
            "momentum": momentum,
            "use_bias": use_bias,
            "epochs": epochs,
            "random_order": random_order,
            "test_size": test_size,
        }

        if dataset == 'iris':
            iris = load_iris()
            X = pd.DataFrame(iris.data)
            y = pd.DataFrame(iris.target)

            network = Network([4, 4, 3], Sigmoid,
                              use_bias=use_bias, use_softmax=True, momentum=momentum)

            wandb.init(project="iris", entity="cladur", config=config)

            if test_size > 0:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, shuffle=True, stratify=y)

                X_train = np.reshape(
                    X_train.to_numpy(), (X_train.shape[0], X_train.shape[1], 1))
                X_test = np.reshape(X_test.to_numpy(),
                                    (X_test.shape[0], X_test.shape[1], 1))
                y_train = classes_to_one_hot(y_train, 3)
                y_train = np.reshape(y_train, (y_train.shape[0], 3, 1))
                y_test = classes_to_one_hot(y_test, 3)
                y_test = np.reshape(y_test, (y_test.shape[0], 3, 1))
            else:
                X_train = np.reshape(X.to_numpy(), (X.shape[0], X.shape[1], 1))
                y_train = classes_to_one_hot(y, 3)
                y_train = np.reshape(y_train, (y_train.shape[0], 3, 1))
                X_test = X_train
                y_test = y_train

            network.train(mse, mse_prime, X_train, y_train, x_test=X_test, y_test=y_test, is_iris=True, epochs=epochs, learning_rate=learning_rate,
                          random_order=random_order, goal_error=goal_error, log_to_file=log_to_file)

            if network_file:
                with open(network_file, 'wb') as f:
                    pickle.dump(network, f)

        if dataset == 'autoencoder':
            wandb.init(project="autoencoder", entity="cladur", config=config)

            X = np.reshape([[1, 0, 0, 0], [0, 1, 0, 0], [
                           0, 0, 1, 0], [0, 0, 0, 1]], (4, 4, 1))
            Y = np.reshape([[1, 0, 0, 0], [0, 1, 0, 0], [
                           0, 0, 1, 0], [0, 0, 0, 1]], (4, 4, 1))

            network = Network([4, 2, 4], Sigmoid, use_bias=use_bias)

            network.train(mse, mse_prime, X, Y, x_test=X, y_test=Y, is_iris=False, epochs=epochs, learning_rate=learning_rate,
                          random_order=random_order, goal_error=goal_error)

            if network_file is not None:
                with open(network_file, 'wb') as f:
                    pickle.dump(network, f)

    def test(self, dataset, network_file):
        if dataset == 'iris':
            with open(network_file, 'rb') as f:
                network = pickle.load(f)

            iris = load_iris()
            X = pd.DataFrame(iris.data)
            y = pd.DataFrame(iris.target)

            X = np.reshape(
                X.to_numpy(), (X.shape[0], X.shape[1], 1))
            y = classes_to_one_hot(y, 3)
            y = np.reshape(y, (y.shape[0], 3, 1))

            network.test(X, y)

        if dataset == 'autoencoder':
            with open(network_file, 'rb') as f:
                network = pickle.load(f)

            X = np.reshape([[1, 0, 0, 0], [0, 1, 0, 0], [
                           0, 0, 1, 0], [0, 0, 0, 1]], (4, 4, 1))
            Y = np.reshape([[1, 0, 0, 0], [0, 1, 0, 0], [
                           0, 0, 1, 0], [0, 0, 0, 1]], (4, 4, 1))

            network.test(X, Y)


if __name__ == '__main__':
    fire.Fire(Program)
