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
