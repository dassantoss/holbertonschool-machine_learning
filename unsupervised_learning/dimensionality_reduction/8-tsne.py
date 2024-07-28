#!/usr/bin/env python3
"""tsne module"""
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) containing the dataset to
        be transformed by t-SNE.
        ndims(int): The new dimensional representation of X.
        idims(int): The intermediate dimensional representation of X after PCA.
        perplexity (float): The perplexity.
        iterations (int): The number of iterations.
        lr (float): The learning rate.

    Returns:
        Y (numpy.ndarray): Array of shape (n, ndim) containing the optimized
        low dimensional transformation of X.
    """
    # Step 1: Reduce the dimensions using PCA
    X_reduced = pca(X, idims)

    # Step 2: Compute P affinities
    P = P_affinities(X_reduced, perplexity=perplexity)

    # Step 3: Initialize the low dimensional representation
    n = X.shape[0]
    Y = np.random.randn(n, ndims)

    # Exaggerate P for the first 100 iterations
    P *= 4

    # Initialize gains and previous gradients for momentum
    gains = np.ones_like(Y)
    dY_prev = np.zeros_like(Y)

    for i in range(1, iterations + 1):
        # Compute gradients and Q affinities
        dY, Q = grads(Y, P)

        # Apply momentum term
        if i < 20:
            momentum = 0.5
        else:
            momentum = 0.8

        # Update Y with the gradient, learning rate, and momentum
        gains = (gains + 0.2) * (np.sign(dY) != np.sign(dY_prev)) + \
                (gains * 0.8) * (np.sign(dY) == np.sign(dY_prev))
        gains = np.maximum(gains, 0.01)
        dY_prev = momentum * dY_prev - lr * gains * dY
        Y += dY_prev

        # Re-center Y
        Y -= np.mean(Y, axis=0)

        # Print cost every 100 iterations
        if i % 100 == 0:
            C = cost(P, Q)
            print(f"Cost at iteration {i}: {C}")

        # Stop exaggerating P after 100 iterations
        if i == 100:
            P /= 4

    return Y
