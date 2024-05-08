#!/usr/bin/env python3
"""
This module defines the NeuralNetwork class for binary classification
with one hidden layer.

Classes:
    NeuralNetwork: Implements a basic neural network.
"""
import numpy as np


class NeuralNetwork:
    """
    Define a neural network with one hidden layer performing binary
    classification.

    Attributes:
        W1 (np.ndarray): Weights vector for the hidden layer.
        b1 (np.ndarray): Bias for the hidden layer.
        A1 (float): Activated output for the hidden layer.
        W2 (np.ndarray): Weights vector for the output neuron.
        b2 (float): Bias for the output neuron.
        A2 (float): Activated output for the output neuron (prediction).
    """

    def __init__(self, nx, nodes):
        """
        Constructor for NeuralNetwork class.

        Args:
            nx (int): Number of input features.
            nodes (int): Number of nodes in the hidden layer.

        Raises:
            TypeError: If nx or nodes are not integers.
            ValueError: If nx or nodes are less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """Getter for private attribute W1."""
        return self.__W1

    @property
    def b1(self):
        """Getter for private attribute b1."""
        return self.__b1

    @property
    def A1(self):
        """Getter for private attribute A1."""
        return self.__A1

    @property
    def W2(self):
        """Getter for private attribute W2."""
        return self.__W2

    @property
    def b2(self):
        """Getter for private attribute b2."""
        return self.__b2

    @property
    def A2(self):
        """Getter for private attribute A2."""
        return self.__A2
