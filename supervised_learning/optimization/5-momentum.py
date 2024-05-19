#!/usr/bin/env python3
"""Module for updating variables using gradient descent with momentum."""
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Update a variable using the gradient descent with momentum optimization
    algorithm.

    Parameters:
    alpha (float): Learning rate.
    beta1 (float): Momentum weight.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of var.
    v (numpy.ndarray): Previous first moment of var.

    Returns:
    numpy.ndarray, numpy.ndarray: Updated variable and the new moment,
        respectively.
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
