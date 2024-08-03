#!/usr/bin/env python3
"""K-means clustering"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means clustering.

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n, d) containing the dataset
                       that will be used for K-means clustering.
                       - n is the number of data points
                       - d is the number of dimensions for each data point
    k (int): A positive integer representing the number of clusters.

    Returns:
    numpy.ndarray: A 2D numpy array of shape (k, d) containing the initialized
                   centroids for each cluster.
                   Returns None on failure
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    return np.random.uniform(min_vals, max_vals, size=(k, X.shape[1]))


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    X (numpy.ndarray): A 2D numpy array of shape (n, d) containing the dataset.
                       - n is the number of data points
                       - d is the number of dimensions for each data point
    k (int): A positive integer representing the number of clusters.
    iterations (int): A positive integer representing the maximum number of
                      iterations that should be performed.

    Returns:
    tuple: (centroids, clss) on success, or (None, None) on failure.
           - centroids is a numpy.ndarray of shape (k, d) containing the
             entroid means for each cluster.
           - clss is a numpy.ndarray of shape (n,) containing the index of the
             cluster in centroids that each data point belongs to.
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
        prev_centroids = np.copy(centroids)

        # Calculate distances and assign clusters
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Calculate new centroids
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            cluster_points = X[clss == j]
            if len(cluster_points) == 0:
                new_centroids[j] = initialize(X, 1)[0]
            else:
                new_centroids[j] = cluster_points.mean(axis=0)

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clss
