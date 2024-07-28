#!/usr/bin/env python3
"""PCA module"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number of
        data points and d is the number of dimensions in each point.
        var (float): Fraction of the variance that the PCA transformation
        should maintain.

    Returns:
        numpy.ndarray: Weights matrix W of shape (d, nd) where nd is the
        new dimensionality of the transformed X.
    """
    # Compute the SVD of the data matrix
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # Compute the total variance explained by the singular values
    cum_variance = np.cumsum(S ** 2) / np.sum(S ** 2)

    # Determine the number of components to keep (indexing starts at 0)
    num_components = np.argmax(cum_variance >= var) + 1

    # Transposed (for shape(d, nd)) top "num_components + 1" rows of Vt
    return Vt[:num_components + 1].T
