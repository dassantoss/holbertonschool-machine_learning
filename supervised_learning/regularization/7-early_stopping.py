#!/usr/bin/env python3
"""
This module contains a function that determines if you should stop
gradient descent early.
"""
import tensorflow as tf


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early.
    Parameters:
        cost (float): Current validation cost of the neural network.
        opt_cost (float): Lowest recorded validation cost of the neural network
        threshold (float): Threshold used for early stopping.
        patience (int): Patience count used for early stopping.
        count (int): Count of how long the threshold has not been met.
    Returns:
        tuple: A boolean indicating whether the network should be stopped early
            followed by the updated count.
    """
    if cost < opt_cost - threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    return False, count
