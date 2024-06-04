#!/usr/bin/env python3
"""Performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    Function that performs a same convolution on grayscale images.
    Args:
    images (numpy.ndarray): A numpy ndarray with shape (m,h,w)
        containing multiple grayscale images.
    kernel (numpy.ndaray): A numpy ndarray with shape (kh,kw)
        containing the kernel for tha convolution.
    Returns:
        numpy.ndarray: The convolved images.
    """
    # Dimensiones
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calcular Padding
    pad_h = kh // 2
    pad_w = kw // 2

    # Aplicar padding
    padded_images = np.pad(images, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           'constant')

    # Inicializo salida
    output = np.zeros((m, h, w))

    # Realizo convolution
    for i in range(h):
        for j in range(w):
            region = padded_images[:, i:i + kh, j:j + kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
