#!/usr/bin/env python3
"""Module for creating a confusion matrix."""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.
    Args:
        labels (numpy.ndarray): A one-hot numpy.ndarray of shape (m, classes)
                            containing the correct labels for each data point.
        logits (numpy.ndarray): A one-hot numpy.ndarray of shape (m, classes)
                            containing the predicted labels.
    Returns:
        numpy.ndarray: A confusion matrix of shape (classes, classes) with row
                        indices representing the correct labels and column
                        indices representing the predicted labels.
    """
    # Convert one-hot encoded labels to integer labels
    true_labels = np.argmax(labels, axis=1)
    pred_labels = np.argmax(logits, axis=1)

    # Number of classes
    classes = labels.shape[1]

    # Initialize the confusion matrix
    confusion_matrix = np.zeros((classes, classes), dtype=int)

    # Populate the confusion matrix
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion_matrix[true_label, pred_label] += 1

    return confusion_matrix
