#!/usr/bin/env python3
"""
Performs a convolution on images using multiple kernels.
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Function that performs a convolution on images using multiple\
    kernels.
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
    m, h, w, _ = images.shape
    kh, kw, _, nc = kernels.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        ph, pw = padding
    elif padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph, pw = 0, 0

    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    output = np.zeros((m, output_h, output_w, nc))

    for k in range(nc):
        for i in range(output_h):
            for j in range(output_w):
                region = padded_images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
                output[:, i, j, k] = np.sum(region * kernels[:, :, :, k],
                                            axis=(1, 2, 3))

    return output
