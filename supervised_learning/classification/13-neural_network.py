#!/usr/bin/env python3
"""
This module defines the NeuralNetwork class for binary classification
with one hidden layer.
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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network using a
        sigmoid activation function.

        Args:
            X (np.ndarray): Numpy array with shape (nx, m) containing the
            input data.

        Returns:
            tuple: The activated outputs (__A1, __A2) of the hidden layer
            and output layer respectively.
        """
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))  # Sigmoid hidden layer.
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))  # Sigmoid output layer.
        return self.__A1, self.__A2

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
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        predictions = (A2 >= 0.5).astype(int)
        return predictions, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform one pass of gradient descen on the neural network.

        Args:
            X (np.ndarray): Input data, shape (nx, m).
            Y (np.ndarray): True labels, shape (1, m).
            A1 (np.ndarray): Output of the hidden layer.
            A2 (np.ndarray): Predicted output.
            alpha (float): Learning rate.

        Updates:
            __W1, __b1: Weights and bias of the hidden layer.
            __W2, __b2: Weights and bias of the output layer.
        """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
