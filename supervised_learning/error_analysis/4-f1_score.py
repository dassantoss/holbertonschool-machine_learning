#!/usr/bin/env python3
"""
Module for calculating F1 score from a confusion matrix.
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct
            labels and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (classes,) containing
        the F1 score of each class.
    """
    # Calculate sensitivity and precision for each class
    sens = sensitivity(confusion)
    prec = precision(confusion)

    # Calculate F1 score
    f1 = 2 * (prec * sens) / (prec + sens)

    return f1
