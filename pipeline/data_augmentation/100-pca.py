#!/usr/bin/env python3
"""
Module for PCA Color Augmentation as described in the AlexNet paper.
"""
import tensorflow as tf


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
    # Normalize the image to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # Reshape the image to (pixel_count, 3)
    flat_image = tf.reshape(image, (-1, 3))

    # Compute the mean and center the image
    mean = tf.reduce_mean(flat_image, axis=0, keepdims=True)
    centered_image = flat_image - mean

    # Compute the covariance matrix
    covariance_matrix = \
        tf.matmul(tf.transpose(centered_image),
                  centered_image) / tf.cast(tf.shape(flat_image)[0],
                                            tf.float32)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = tf.linalg.eigh(covariance_matrix)

    # Adjust colors using eigenvalues and alphas
    delta = tf.matmul(eigenvectors, tf.reshape(eigenvalues * alphas, [-1, 1]))
    delta = tf.transpose(delta)
    delta = tf.broadcast_to(delta, tf.shape(centered_image))
    augmented_image = centered_image + delta + mean

    # Clip values to [0, 1] and reshape back to original shape
    augmented_image = tf.clip_by_value(augmented_image, 0, 1)
    augmented_image = tf.reshape(augmented_image, tf.shape(image))

    # Convert back to tensor and scale to [0, 255]
    return tf.cast(augmented_image * 255, tf.uint8)
