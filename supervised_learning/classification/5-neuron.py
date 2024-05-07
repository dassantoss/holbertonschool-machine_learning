#!/usr/bin/env python3
"""
Module implementing the Neuron class for binary classification tasks.

Classes:
    Neuron: Implements a single neuron with methods for initialization,
    forward propagation, cost, valuate and gradient descent.

"""
import numpy as np


class Neuron:
    """
    Implements a single neuron for binary classification.

    Attributes:
        __W (numpy.ndarray): The weights of the neuron.
        __b (float): The bias of the neuron.
        __A (float): The activated output of the neuron.

    """

    def __init__(self, nx):
        """
        Initializes a Neuron object.

        Parameters:
            nx (int): The number of input features to the neuron.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        elif nx < 1:
            raise ValueError('nx must be a positive integer')

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    def forward_prop(self, X):
        """
        Performs forward propagation using the sigmoid activation function.

        Args:
            X (numpy.ndarray): Input data of shape (nx, m).

        Returns:
            numpy.ndarray: The activated output.
        """
        z = np.dot(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):
        """
        Calculate the cost of the model using logistic regression.

        Args:
            Y (numpy.ndarray): Correct labels for the input data, shape (1,m).
            A (numpy.ndarray): Activated output of the neuron, shampe (1,m).

        Returns:
            float: The logistic regression cost.
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A)
                                 + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluate the neuron's predictions.

        Args:
            X (numpy.ndarray): Input data, shape (nx, m).
            Y (numpy.ndarray): Correct labels, shape (1, m).

        Returns:
            tuple: (numpy.ndarray, float)
                - Predictions for each example (1 if A >= 0.5 else 0).
                - The cost of the network.
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = (A >= 0.5).astype(int)
        return predictions, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Perform one pass of gradient descent on the neuron to update
        weights and bias.

        Args:
            X (numpy.ndarray): Input data, shape (nx, m).
            Y (numpy.ndarray): Correct labels, shape (1, m).
            A (numpy.ndarray): Activated output of the neuron.
        """
        m = Y.shape[1]
        dz = A - Y  # Compute error vector
        dW = np.dot(dz, X.T) / m  # Derivative of the cost respect to W
        db = np.sum(dz) / m  # Derivative of the cost with respect to b
        self.__W -= alpha * dW  # Update Weights
        self.__b -= alpha * db  # Updated Bias

    @property
    def W(self):
        """
        Getter method for the weights of the neuron.

        Returns:
            numpy.ndarray: The weights of the neuron.
        """
        return self.__W

    @property
    def b(self):
        """
        Getter method for the bias of the neuron.

        Returns:
            float: The bias of the neuron.
        """
        return self.__b

    @property
    def A(self):
        """
        Getter method for the activated output of the neuron.

        Returns:
            float: The activated output of the neuron.
        """
        return self.__A
