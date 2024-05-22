#!/usr/bin/env python3
"""
Module for calculating precision from a confusion matrix.
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct labels
            and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (classes,) containing the
        precision of each class.
    """
    # True Positives (TP) are the diagonal elements of the confusion matrix
    TP = np.diag(confusion)

    # False Positives (FP) are the sum of the column elements minus the TP
    FP = np.sum(confusion, axis=0) - TP

    # Precision is TP / (TP + FP)
    precision = TP / (TP + FP)

    return precision
