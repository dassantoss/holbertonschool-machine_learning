#!/usr/bin/env python3
"""P_init module"""
import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to calculate the P affinities in t-SNE.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) where n is the number of
        data points and d is the number of dimensions in each point.
        perplexity (float): The perplexity that all Gaussian distributions
        should have.

    Returns:
        D (numpy.ndarray): Array of shape (n, n) that calculates the squared
        pairwise distance between data points.
        P (numpy.ndarray): Array of shape (n, n) initialized to all 0's that
        will contain the P affinities.
        betas (numpy.ndarray): Array of shape (n, 1) initialized to all 1's
        that will contain all of the beta values.
        H (float): Shannon entropy for perplexity with a base of 2.
    """
    n, d = X.shape

    # Compute the squared pairwise distance matrix D
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)

    # Initialize P, betas, and H
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return D, P, betas, H
