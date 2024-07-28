#!/usr/bin/env python3
"""P_affinities module"""
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set.

    Args:
        X (numpy.ndarray): Dataset of shape (n, d) containing the dataset
                           to be transformed by t-SNE.
        tol (float): Maximum tolerance allowed for the difference in Shannon
        entropy.
        perplexity (float): The perplexity that all Gaussian distributions
        should have.

    Returns:
        numpy.ndarray: Array of shape (n, n) containing the symmetric P
        affinities.
    """
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        betas[i] = 1.0
        betamin = None
        betamax = None
        Di = np.delete(D[i], i)
        Hdiff = 1
        tries = 0

        while abs(Hdiff) > tol and tries < 50:
            Hi, Pi = HP(Di, betas[i])
            Hdiff = Hi - H
            if Hdiff > 0:
                betamin = betas[i]
                if betamax is None:
                    betas[i] *= 2.0
                else:
                    betas[i] = (betas[i] + betamax) / 2.0
            else:
                betamax = betas[i]
                if betamin is None:
                    betas[i] /= 2.0
                else:
                    betas[i] = (betas[i] + betamin) / 2.0
            tries += 1

        P[i, np.arange(n) != i] = Pi

    # Symmetrize P and normalize
    P = (P + P.T) / (2 * n)
    return P
