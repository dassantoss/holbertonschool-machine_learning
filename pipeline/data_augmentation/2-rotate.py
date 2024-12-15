#!/usr/bin/env python3
"""
Module that contains a function that rotates an image by 90 degrees
counter-clockwise
"""
import tensorflow as tf


def rotate_image(image):
    """
    Function that rotates an image by 90 degrees counter-clockwise

    Args:
        image: 3D tf.Tensor containing the image to rotate

    Returns:
        The rotated image
    """
    return tf.image.rot90(image)
