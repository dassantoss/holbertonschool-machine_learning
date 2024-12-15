#!/usr/bin/env python3
"""
Module that contains a function that changes the hue of an image
"""
import tensorflow as tf


def change_hue(image, delta):
    """
    Function that changes the hue of an image

    Args:
        image: 3D tf.Tensor containing the image to change
        delta: amount the hue should change

    Returns:
        The hue-adjusted image
    """
    return tf.image.adjust_hue(image, delta)
