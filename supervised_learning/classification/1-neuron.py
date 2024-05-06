#!/usr/bin/env python3
"""
This module defines the Neuron class for binary classification tasks.

Classes:
    Neuron: Implements a single neuron with methods for initialization
    with private attributes.
"""
import numpy as np


class Neuron:
    """Defines a single neuron performing binary classification with
    private attributes."""

    def __init__(self, nx):
        """
        Constructor for Neuron class.

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

    @property
    def W(self):
        """Getter for private attribute W."""
        return self.__W

    @property
    def b(self):
        """Getter for private attribute b."""
        return self.__b

    @property
    def A(self):
        """Getter for private attribute A."""
        return self.__A
