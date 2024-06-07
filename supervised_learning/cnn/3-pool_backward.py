#!/usr/bin/env python3
"""
Performs back propagation over a pooling layer
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Function that performs back propagation over a pooling layer

    Args:
        dA (numpy.ndarray): of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the output of the
            pooling layer
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c) containing
            the output of the previous layer
        kernel_shape (tuple): (kh, kw) containing the size of the kernel
            for the pooling
            kh is the kernel height
            kw is the kernel width
        stride (tuple): (sh, sw) containing the strides
            sh is the stride for the height
            sw is the stride for the width
        mode (str): string that is either 'max' or 'avg', indicating the
            type of pooling

    Returns:
        dA_prev (numpy.ndarray): Partial derivatives with respect to the
            previous layer
    """
    # Setup stride, matrixes and kernel dimensions
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    m, h_new, w_new, c_new = dA.shape

    # Initialize dA_prev
    dA_prev = np.zeros_like(A_prev)

    for i in range(h_new):
        for j in range(w_new):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            if mode == 'max':
                # Backward Max Pooling
                for k in range(c):
                    A_slice = A_prev[:, vert_start:vert_end,
                                     horiz_start:horiz_end, k]
                    mask = (A_slice == np.max(A_slice, axis=(1, 2),
                                              keepdims=True))
                    dA_prev[:, vert_start:vert_end, horiz_start:horiz_end,
                            k] += (mask * dA[:, i, j, k][:, np.newaxis,
                                                         np.newaxis])
            elif mode == 'avg':
                # Backward Average Pooling
                da = dA[:, i, j, :, np.newaxis, np.newaxis]
                dA_prev[:, vert_start:vert_end, horiz_start:horiz_end, :] += (
                    da / (kh * kw))

    return dA_prev
