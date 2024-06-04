#!/usr/bin/env python3
"""Performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images.
    Args:
    images (numpy.ndarray): A numpy ndarray with shape (m,h,w)
        containing multiple grayscale images.
    kernel (numpy.ndaray): A numpy ndarray with shape (kh,kw)
        containing the kernel for tha convolution.
    Returns:
        numpy.ndarray: The convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    output_h = h - kh + 1
    output_w = w - kw + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            region = images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
