#!/usr/bin/env python3
"""K-means clustering"""
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
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(min_vals, max_vals, (k, X.shape[1]))
    return centroids


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset

    Parameters:
    X (numpy.ndarray): The dataset of shape (n, d)
    k (int): The number of clusters
    iterations (int): The maximum number of iterations

    Returns:
    numpy.ndarray: Centroid means for each cluster
    numpy.ndarray: Index of the cluster each data point belongs to
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        new_centroids = np.array([
            X[clss == j].mean(axis=0) if np.any(clss == j)
            else initialize(X, 1)[0]
            for j in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clss
