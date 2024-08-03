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
    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None
    if X.ndim != 2 or k <= 0:
        return None

    n, d = X.shape
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)

    centroids = np.random.uniform(min_vals, max_vals, (k, d))
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
    if not isinstance(X, np.ndarray) or not isinstance(k, int) or \
       not isinstance(iterations, int):
        return None, None
    if X.ndim != 2 or k <= 0 or iterations <= 0:
        return None, None

    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    for _ in range(iterations):
        # Assign each data point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        # Calculate new centroids
        new_centroids = np.array([
            X[clss == j].mean(axis=0) if np.any(clss == j)
            else initialize(X, 1)[0]
            for j in range(k)
        ])

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, clss
