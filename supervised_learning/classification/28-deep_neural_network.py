#!/usr/bin/env python3
"""
This module defines the DeepNeuralNetwork.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


class DeepNeuralNetwork:
    """
    Define a deep neural network performing multiclass classification.

    Attributes:
        L (int): Number of layers in the neural network.
        cache (dict): A dictionary to hold all intermediary values of
            the network.
        weights (dict): A dictionary to hold all weights and biases of
            the network.
    """

    def __init__(self, nx, layers, activation='sig'):
        """
        Constructor for DeepNeuralNetwork class.

        Args:
            nx (int): Number of input features.
            layers (list of int): Number of nodes in each layer of the
                network.
            activation (str): Activation function to be used; 'sig' for
                sigmoid or 'tanh' for tanh.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list or an empty list.
            TypeError: If elements in layers are not all positive integers.
            ValueError: If activation is not 'sig' or 'tanh'.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        # Initialize weights and biases
        for i in range(1, self.__L + 1):
            layer_size = layers[i - 1]
            prev_layer_size = nx if i == 1 else layers[i - 2]

            # He initialization weights
            self.__weights[f'W{i}'] = \
                np.random.randn(layer_size, prev_layer_size) \
                * np.sqrt(2 / prev_layer_size)
            self.__weights[f'b{i}'] = np.zeros((layer_size, 1))

    @property
    def L(self):
        """Getter for the number of layers."""
        return self.__L

    @property
    def cache(self):
        """Getter for the cache dictionary containing intermediary values
            of the network."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the dictionary of weights and biases of the
            network."""
        return self.__weights

    @property
    def activation(self):
        """ Getter para la función de activación """
        return self.__activation

    def forward_prop(self, X):
        """
        Calculate the forward propagation of the deep neural network.

        Args:
            X (numpy.ndarray): Data input (nx, m).

        Returns:
            numpy.ndarray: The output of the last layer.
            dict: A cache containing all the intermediary values of
            the network.
        """
        self.__cache['A0'] = X
        A = X
        for i in range(1, self.__L + 1):
            prev_A = self.__cache[f"A{i - 1}"]
            Z = np.matmul(self.__weights[f"W{i}"], prev_A) + self.__weights[f"b{i}"]
            if i < self.__L:  # Aplicación de tanh o sig
                if self.__activation == 'sig':
                    A = 1 / (1 + np.exp(-Z))
                else:  # tanh
                    A = np.tanh(Z)
            else:  # Softmax en la última capa
                exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
                A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
            self.__cache[f"A{i}"] = A
        return self.__cache[f"A{self.__L}"], self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost of the model using cross-entropy loss.

        Args:
            Y (np.ndarray): Correct labels for the input data, shape
                (classes, m).
            A (np.ndarray): Activated output of the last layer, shape
                (classes, m).

        Returns:
            float: The cross-entropy cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.

        Args:
            X (np.ndarray): Numpy array with shape (nx, m)
                containing the input data.
            Y (np.ndarray): Numpy array with shape (classes, m)
                containing the correct labels for the input data.

        Returns:
            tuple: A tuple containing:
                - Numpy array with shape (classes, m) containing the
                    predicted labels for each example.
                - The cost of the network.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.eye(A.shape[0])[np.argmax(A, axis=0)].T
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the deep neural network.

        Args:
            Y (np.ndarray): True labels, shape (classes, m) where
                'classes' is the number of output classes.
            cache (dict): Dictionary containing all intermediary
                values of the network.
            alpha (float): Learning rate.

        Updates:
            __weights (dict): Private dictionary storing the weights
                and biases, adjusted by gradient descent.
        """
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y  # Difference at the output layer

        for i in reversed(range(1, self.__L + 1)):
            A_prev = cache[f'A{i-1}'] if i > 1 else cache['A0']

            # Compute gradients
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 1:
                W_current = self.__weights[f'W{i}']
                if self.__activation == 'sig':
                    dZ = np.dot(W_current.T, dZ) * (A_prev * (1 - A_prev))
                elif self.__activation == 'tanh':
                    dZ = np.dot(W_current.T, dZ) * (1 - np.square(np.tanh(A_prev)))

            # Update weights and biases after preparing dZ
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f"b{i}"] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the deep neural network by updating the private attributes
        __weights and __cache.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).
            Y (numpy.ndarray): True labels for the data of shape (1, m).
            iterations (int): Number of iterations to train.
            alpha (float): Learning rate.
            verbose (bool): Whether to print training information.
            graph (bool): Whether to plot the training cost.
            step (int): Step size for printing and plotting.

        Returns:
            tuple: (numpy.ndarray, float)
                - Predictions after training.
                - Cost after training.
        """
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        costs = []
        count = []
        for i in range(iterations + 1):
            A, _ = self.forward_prop(X)

            if i != iterations:
                self.gradient_descent(Y, self.cache, alpha)

            cost = self.cost(Y, A)
            costs.append(cost)
            count.append(i)

            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))

        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Save the instance object to a file in pickle format.

        Args:
            filename (str): The file to which the object should be saved.
        """
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a pickled DeepNeuralNetwork object.

        Args:
            filename (str): The file from which the object should be loaded.

        Returns:
            DeepNeuralNetwork: The loaded object, or None if the file doesn’t
            exist.
        """
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None
