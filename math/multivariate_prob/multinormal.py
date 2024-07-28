#!/usr/bin/env python3
"""Module to create a class MultiNormal that represents a Multivariate
Normal distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Initializes the MultiNormal instance with the given data set.
        Parameters:
        - data (numpy.ndarray): Data set of shape (d, n) where n is the
        number of data points
                                and d is the number of dimensions in each
                                data point.

        Raises:
        - TypeError: If data is not a 2D numpy.ndarray.
        - ValueError: If data does not contain multiple data points.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Calculate the mean of the data set
        self.mean = np.mean(data, axis=1, keepdims=True)

        # Calculate the covariance matrix
        X_centered = data - self.mean
        self.cov = np.dot(X_centered, X_centered.T) / (n - 1)
