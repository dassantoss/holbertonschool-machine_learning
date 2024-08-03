#!/usr/bin/env python3
"""Initialize K-means centroids"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    Parameters:
    X (numpy.ndarray): The dataset of shape (n, d)
    k (int): The number of clusters

    Returns:
    numpy.ndarray: The initialized centroids of shape (k, d) or None on failure
    """
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None
    if X.ndim != 2 or k <= 0:
        return None

    n, d = X.shape
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(min_vals, max_vals, (k, d))
    return centroids
