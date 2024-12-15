#!/usr/bin/env python3
"""
Module that contains a function that randomly adjusts the contrast of an image
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Function that randomly adjusts the contrast of an image

    Args:
        image: A 3D tf.Tensor representing the input image to adjust the
               contrast
        lower: A float representing the lower bound of the random contrast
               factor range
        upper: A float representing the upper bound of the random contrast
               factor range

    Returns:
        The contrast-adjusted image
    """
    return tf.image.random_contrast(image, lower, upper)
