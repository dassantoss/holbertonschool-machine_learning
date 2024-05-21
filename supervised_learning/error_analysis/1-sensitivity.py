#!/usr/bin/env python3
"""
Module for calculating sensitivity from a confusion matrix.
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.
    Args:
        confusion (numpy.ndarray): A confusion numpy.ndarray of shape
        (classes, classes)
        where row indices represent the correct labels and
        column indices represent the predicted labels.

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (classes,) containing
        the sensitivity of each class.
    """
    # True Positives (TP) are the diagonal elements of the confusion matrix
    TP = np.diag(confusion)

    # False Negatives (FN) are the sum of the row elements minus the TP
    FN = np.sum(confusion, axis=1) - TP

    # Sensitivity is TP / (TP + FN)
    sensitivity = TP / (TP + FN)

    return sensitivity
