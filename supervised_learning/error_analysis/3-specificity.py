"""
Module for calculating specificity from a confusion matrix.
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A confusion numpy.ndarray of shape
            (classes, classes) where row indices represent the correct labels
            and column indices represent the predicted labels.

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (classes,) containing the
        specificity of each class.
    """
    # True Positives (TP) are the diagonal elements of the confusion matrix
    TP = np.diag(confusion)

    # False Positives (FP) are the sum of the column elements minus the TP
    FP = np.sum(confusion, axis=0) - TP

    # False Negativies (FN) are the sum of the row elements minus the TP
    FN = np.sum(confusion, axis=1) - TP

    # True negatives (TN) are the total sum minus TP, FP, and FN
    TN = np.sum(confusion) - (TP + FP + FN)

    # Specificity is TN / (TN + FP)
    specificity = TN / (TN + FP)

    return specificity
