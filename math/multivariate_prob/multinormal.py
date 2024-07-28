#!/usr/bin/env python3
"""Module to create a class MultiNormal that represents a Multivariate Normal
distribution."""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution."""

    def __init__(self, data):
        """
        Initializes the MultiNormal instance with the given data set.

        Parameters:
        - data (numpy.ndarray): Data set of shape (d, n) where n is the number
        of data points and d is the number of dimensions in each data point.

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

    def pdf(self, x):
        """
        Calculates the PDF at a data point.

        Parameters:
        - x (numpy.ndarray): Data point of shape (d, 1) where d is the number
        of dimensions.
        Returns:
        - float: The value of the PDF at the data point.

        Raises:
        - TypeError: If x is not a numpy.ndarray.
        - ValueError: If x does not have the shape (d, 1).
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d, _ = self.mean.shape
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        # Calculate the PDF using the multivariate normal distribution formula
        det_cov = np.linalg.det(self.cov)
        inv_cov = np.linalg.inv(self.cov)
        norm_factor = 1 / np.sqrt(((2 * np.pi) ** d) * det_cov)

        x_centered = x - self.mean
        exponent = -0.5 * np.dot(np.dot(x_centered.T, inv_cov), x_centered)

        pdf_value = norm_factor * np.exp(exponent)

        return pdf_value[0, 0]
