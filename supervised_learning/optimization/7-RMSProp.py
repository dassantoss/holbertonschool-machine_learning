#!/usr/bin/env python3
"""Module for updating variables using the RMSProp optimization algorithm."""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Update a variable using the RMSProp optimization algorithm.

    Parameters:
    alpha (float): The learning rate.
    beta2 (float): The RMSProp weight.
    epsilon (float): A small number to avoid division by zero.
    var (numpy.ndarray): The variable to be updated.
    grad (numpy.ndarray): The gradient of var.
    s (numpy.ndarray): The previous second moment of var.

    Returns:
    numpy.ndarray, numpy.ndarray: The updated variable and the new moment,
    respectively.
    """
    s = beta2 * s + (1 - beta2) * grad**2
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
