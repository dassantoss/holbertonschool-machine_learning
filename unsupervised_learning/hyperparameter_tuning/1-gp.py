#!/usr/bin/env python3
"""Gaussian Process module for noiseless 1D GP."""
import numpy as np


class GaussianProcess:
    """Represents a noiseless 1D Gaussian process."""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):  # noqa: E741
        """
        Class constructor.

        Args:
            X_init (numpy.ndarray): Inputs already sampled with the black-box
                                    function, shape (t, 1).
            Y_init (numpy.ndarray): Outputs of the black-box function for each
                                    input in X_init, shape (t, 1).
            l (float): Length parameter for the kernel.
            sigma_f (float): Standard deviation given to the output of the
                             black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        # Calculate the initial covariance matrix using the kernel function
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices using the
        RBF kernel.

        Args:
            X1 (numpy.ndarray): First set of inputs, shape (m, 1).
            X2 (numpy.ndarray): Second set of inputs, shape (n, 1).

        Returns:
            numpy.ndarray: Covariance kernel matrix, shape (m, n).
        """
        # Calculate the squared Euclidean distance between each pair of points
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        # Apply the RBF kernel formula
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation of points in a Gaussian
        process.

        Args:
            X_s (numpy.ndarray): Points at which to predict, shape (s, 1).

        Returns:
            mu (numpy.ndarray): Mean of predictions, shape (s,).
            sigma (numpy.ndarray): Variance of predictions, shape (s,).
        """
        # Calculate the covariance between X_s and the initial points X_init
        K_s = self.kernel(self.X, X_s)
        # Calculate the covariance between X_s points
        K_ss = self.kernel(X_s, X_s)
        # Inverse of the initial covariance matrix K
        K_inv = np.linalg.inv(self.K)

        # Calculate the mean of the predictive distribution
        mu = K_s.T.dot(K_inv).dot(self.Y).reshape(-1)

        # Calculate the covariance (variance) of the predictive distribution
        sigma = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(sigma)  # Extract the diagonal elements (variances)

        return mu, sigma
