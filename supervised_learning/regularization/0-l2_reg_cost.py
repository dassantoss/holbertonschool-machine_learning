#!/usr/bin/env python3
"""
This module contains a function that calculates the cost
of the neural network with L2 regularization.
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.
    Args:
        cost (float): Cost of the network without L2 regularization.
        lambtha (float): Ragularization parameter.
        weights (dict): Dicctionary of the weights and biases.
        L (int): Number of layers in the neural network.
        m (int): Number of data points used.
    Returns:
        float: Cost of the network accounting for L2 regularization.
    """
    l2_cost = 0
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_cost += np.sum(np.square(W))

    l2_cost = (lambtha / (2 * m)) * l2_cost
    total_cost = cost + l2_cost

    return total_cost
