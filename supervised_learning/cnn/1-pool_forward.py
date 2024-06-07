#!/usr/bin/env python3
"""
Performs forward propagation over a pooling layer
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs forward propagation over a pooling layer

    Args:
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        kernel_shape (tuple): (kh, kw) containing the size of the kernel
            kh is the kernel height
            kw is the kernel width
        stride (tuple): (sh, sw) containing the strides
            sh is the stride for the height
            sw is the stride for the width
        mode (str): string that is either 'max' or 'avg', indicating
            the type of pooling

    Returns:
        numpy.ndarray: Output of the pooling layer
    """
    # Setup stride, matrixes and kernel dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # Calculate output dimensions
    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1

    # Initialize pooling output array
    pooled = np.zeros((m, output_h, output_w, c_prev))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from input
            region = A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            if mode == 'max':
                # Apply max pooling
                pooled[:, i, j, :] = np.max(region, axis=(1, 2))
            elif mode == 'avg':
                # Apply average pooling
                pooled[:, i, j, :] = np.mean(region, axis=(1, 2))

    return pooled
