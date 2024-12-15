#!/usr/bin/env python3
"""
Module for PCA Color Augmentation as described in the AlexNet paper.
"""
import tensorflow as tf
import numpy as np


def pca_color(image, alphas):
    """
    Performs PCA color augmentation on an image.

    Args:
        image (tf.Tensor): A 3D tensor containing the image to augment.
        alphas (tuple): A tuple of length 3 containing the amount each
                        channel should change.

    Returns:
        tf.Tensor: The augmented image.
    """
    # Convert image to numpy array and normalize to [0, 1]
    image_np = tf.cast(image, tf.float32).numpy() / 255.0

    # Reshape to a 2D array (pixel_count, channels)
    flat_image = image_np.reshape(-1, 3)

    # Compute the covariance matrix
    covariance_matrix = np.cov(flat_image, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Adjust colors using eigenvalues and alphas
    delta = eigenvectors @ (eigenvalues * alphas)
    augmented_image = flat_image + delta

    # Clip values to [0, 1] and reshape back to original shape
    augmented_image = np.clip(augmented_image, 0, 1).reshape(image_np.shape)

    # Convert back to tensor and scale to [0, 255]
    return tf.convert_to_tensor(augmented_image * 255, dtype=tf.uint8)
