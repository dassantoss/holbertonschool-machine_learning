#!/usr/bin/env python3
"""
Performs a same convolution on grayscale images with padding
and stride
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Function that Performs a same convolution on grayscale images with
    padding and stride.
    Args:
    images (numpy.ndarray): A numpy ndarray with shape (m,h,w)
        containing multiple grayscale images.
    kernel (numpy.ndaray): A numpy ndarray with shape (kh,kw)
        containing the kernel for tha convolution.
    padding (tuple): Either tuple of (ph, pw), 'same' or 'valid'.
    stride (tuple): A tuple of (sh, sw) representing the stride for
        height and witdth.
    Returns:
        numpy.ndarray: The convolved images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    padded_h = h + 2 * ph
    padded_w = w + 2 * pw
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            x = i * sh
            y = j * sw
            region = padded_images[:, x:x+kh, y:y+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
