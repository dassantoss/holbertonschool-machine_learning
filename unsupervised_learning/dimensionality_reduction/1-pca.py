#!/usr/bin/env python3
"""PCA v2 module"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset to reduce it to a specified number of dimensions.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number of
        data points and d is the number of dimensions in each point.
        ndim (int): New dimensionality of the transformed X.

    Returns:
        numpy.ndarray: Transformed version of X with shape (n, ndim).
    """
    # Compute the mean of X
    X_mean = X - np.mean(X, axis=0)

    # Compute the SVD of the data matrix
    U, S, Vt = np.linalg.svd(X_mean, full_matrices=False)

    # Select the top ndim components from Vt
    W = Vt[:ndim].T

    # Transform the data to the new dimensionality
    T = np.matmul(X_mean, W)

    return T
