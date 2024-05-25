#!/usr/bin/env python3
"""
This module contains a function that updates the weights of a neural
network with Dropout regularization using gradient descent.
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization using
    gradient descent.
    Parameters:
    Y (numpy.ndarray): One-hot numpy.ndarray of shape (classes,m) that contains
        the correct labels for the data.
    weights (dict): Dictionary of the weights and biases..
    cache (dict): Dictionary of the outputs and dropout masks of each layer.
    alpha (float): Learning rate.
    keep_prob (float): Probability that a node will be kept.
    L (int): Number of layers of the network.
    """
    m = Y.shape[1]
    A_prev = cache['A' + str(L - 1)]
    A_curr = cache['A' + str(L)]
    dZ = A_curr - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]

        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            dA_prev = np.dot(W.T, dZ)
            D = cache['D' + str(layer - 1)]
            dA_prev = dA_prev * D
            dA_prev = dA_prev / keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)  # Derivative of tanh

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
