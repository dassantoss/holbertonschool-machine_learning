#!/usr/bin/env python3
"""Module that contains a function that performs a random crop of an image"""
import tensorflow as tf


def crop_image(image, size):
    """
    Function that performs a random crop of an image

    Args:
        image: 3D tf.Tensor containing the image to crop
        size: tuple containing the size of the crop

    Returns:
        The cropped image
    """
    return tf.image.random_crop(image, size)
