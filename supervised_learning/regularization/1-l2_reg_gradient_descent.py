#!/usr/bin/env python3
"""
This module contains a fuction that update the weights and biases of a
neural network using gradient descent withn L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.
    Parameters:
        Y (numpy.ndarray): One-hot numpy.ndarray of shape (classes, m) that
            contains the correct labels for the data.
        weights (dict): Dictionary of the weights and biases..
        cache (dict): Dictionary of the outputs of each layer.
        alpha (float): Learning rate.
        lambtha (float): L2 regularization parameter.
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

        dW = (np.dot(dZ, A_prev.T) + lambtha * W) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if layer > 1:
            dA_prev = np.dot(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)  # tanh derivative

        weights['W' + str(layer)] = W - alpha * dW
        weights['b' + str(layer)] = b - alpha * db
