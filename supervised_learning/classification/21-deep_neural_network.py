#!/usr/bin/env python3
"""
This module defines the DeepNeuralNetwork class for binary classification.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    Define a deep neural network performing binary classification.

    Attributes:
        L (int): Number of layers in the neural network.
        cache (dict): A dictionary to hold all intermediary values of
            the network.
        weights (dict): A dictionary to hold all weights and biases of
            the network.
    """

    def __init__(self, nx, layers):
        """
        Constructor for DeepNeuralNetwork class.

        Args:
            nx (int): Number of input features.
            layers (list of int): Number of nodes in each layer of the
                network.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
            TypeError: If layers is not a list or an empty list.
            TypeError: If elements in layers are not all positive integers.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: isinstance(x, int) and x > 0, layers)):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)  # Number of layers
        self.__cache = {}  # Dictionary to store forward propagation values
        self.__weights = {}  # Dictionary to store weights and biases

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
        """Getter for the number of cache."""
        return self.__cache

    @property
    def weights(self):
        """Getter for the number of weights."""
        return self.__weights

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
            W = self.__weights[f'W{i}']
            b = self.__weights[f'b{i}']
            Z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache[f'A{i}'] = A

        return A, self.__cache

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression.

        Args:
            Y (np.ndarray): Correct labels for the input data, shape(1,m).
            A (np.ndarray): Activated output of the neuron for each
            example, shape(1,m).

        Returns:
            float: The logistic regression cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A) + (1 - Y)
                                 * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neural network's predictions.

        Args:
            X (np.ndarray): Numpy array with shape (nx, m) containing the
            input data.
            Y (np.ndarray): Numpy array with shape (1, m) containing the
            correct labels for the input data.

        Returns:
            tuple: A tuple containing:
                - Numpy array with shape (1, m) containing the predicted
                labels for each example.
                - The cost of the network.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Performs one pass of gradient descent on the deep neural network.

        Args:
            Y (np.ndarray): True labels, shape (1, m).
            cache (dict): Dictionary containing all intermediary values
            of the network.
            alpha (float): Learning rate.

        Updates:
            __weights (dict): Private dictionary storing the weights and
            biases, adjusted by gradient descent.
        """
        m = Y.shape[1]
        dZ = cache[f"A{self.__L}"] - Y  # Difference at the output

        for i in reversed(range(1, self.__L + 1)):
            A_prev = cache[f'A{i-1}']

            # Compute gradients
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # Update weights and biases
            self.__weights[f'W{i}'] -= alpha * dW
            self.__weights[f'b{i}'] -= alpha * db

            if i > 1:
                # Calculate dZ next layer
                W_current = self.__weights[f'W{i}']
                dZ = np.dot(W_current.T, dZ) * (A_prev * (1 - A_prev)) 
