#!/usr/bin/env python3
"""Bayesian Optimization module for noiseless 1D Gaussian process."""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Performs Bayesian optimization on a noiseless 1D Gaussian process."""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor.

        Args:
            f (function): The black-box function to be optimized.
            X_init (numpy.ndarray): Initial input samples, shape (t, 1).
            Y_init (numpy.ndarray): Initial output samples, shape (t, 1).
            bounds (tuple): Bounds of the space in which to look for the
                            optimal point, (min, max).
            ac_samples (int): Number of samples to analyze during acquisition.
            l (float): Length parameter for the kernel.
            sigma_f (float): Standard deviation for the GP model.
            xsi (float): Exploration-exploitation factor for acquisition.
            minimize (bool): Whether to minimize (True) or maximize (False) the
            function.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        self.xsi = xsi
        self.minimize = minimize

        # Generate evenly spaced acquisition sample points within the bounds
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)

    def acquisition(self):
        """
        Calculates the next best sample location using the Expected Improvement
        acquisition function.

        Returns:
            X_next (numpy.ndarray): Next best sample point, shape (1,).
            EI (numpy.ndarray): Expected improvement for each sample, shape
            (ac_samples,).
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_opt = np.min(self.gp.Y)
            imp = Y_opt - mu - self.xsi
        else:
            Y_opt = np.max(self.gp.Y)
            imp = mu - Y_opt - self.xsi

        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function.

        Args:
            iterations (int): Maximum number of iterations to perform.

        Returns:
            X_opt (numpy.ndarray): Optimal point, shape (1,).
            Y_opt (numpy.ndarray): Optimal function value, shape (1,).
        """
        for _ in range(iterations):
            # Find next best point to sample
            X_next, _ = self.acquisition()

            # Stop early if the next point has already been sampled
            if np.any(np.isclose(X_next, self.gp.X)):
                break

            # Sample function at the proposed point
            Y_next = self.f(X_next)

            # Update our Gaussian Process with the new sample
            self.gp.update(X_next, Y_next)

        # Calc. optimal point and its function value
        if self.minimize:
            optimal_idx = np.argmin(self.gp.Y)
        else:
            optimal_idx = np.argmax(self.gp.Y)

        self.gp.X = self.gp.X[:-1, :]
        X_opt = self.gp.X[optimal_idx]
        Y_opt = self.gp.Y[optimal_idx]

        return X_opt, Y_opt
