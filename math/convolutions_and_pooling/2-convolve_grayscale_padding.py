#!/usr/bin/env python3
"""Performs a same convolution on grayscale images with custom padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Function that performs a same convolution on grayscale images with
    custom padding.
    Args:
    images (numpy.ndarray): A numpy ndarray with shape (m,h,w)
        containing multiple grayscale images.
    kernel (numpy.ndaray): A numpy ndarray with shape (kh,kw)
        containing the kernel for tha convolution.
    padding (tuple): A tuple of (ph, pw) representing the padding for
        height and width of the image.
    Returns:
        numpy.ndarray: The convolved images.
    """
    # Dimensiones
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # Dimensiones de output
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1

    # Aplicar padding
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # Inicializo salida
    output = np.zeros((m, output_h, output_w))

    # Realizo convolution
    for i in range(output_h):
        for j in range(output_w):
            region = padded_images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
