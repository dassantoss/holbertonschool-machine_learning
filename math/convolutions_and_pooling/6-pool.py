#!/usr/bin/env python3
"""
Performs performs pooling on images.
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs pooling on images.
    Args:
    images (numpy.ndarray): A numpy ndarray with shape (m,h,w,c)
        containing multiple images.
    kernel (numpy.ndaray): A numpy ndarray with shape (kh,kw)
        containing the kernel shape.
    padding (tuple): Either tuple of (ph, pw), 'same' or 'valid'.
    stride (tuple): A tuple of (sh, sw) representing the stride
        for height and width.
    mode (str): Indicates the type of pooling. 'max'
        indicates max pooling, 'avg' indicates average
        pooling.
    Returns:
        numpy.ndarray: The convolved images.
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1

    # Choose pooling function based on given mode
    if mode == 'max':
        pooling_func = np.max
    elif mode == 'avg':
        pooling_func = np.mean

    output = np.zeros((m, output_h, output_w, c))

    for i in range(output_h):
        for j in range(output_w):
            region = images[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            output[:, i, j, :] = pooling_func(region, axis=(1, 2))

    return output
