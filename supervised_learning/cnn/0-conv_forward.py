#!/usr/bin/env python3
"""
Performs forward propagation over a CNN
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Function that performs forward propagation over a CNN

    Args:
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W (numpy.ndarray): of shape (kh, kw, c_prev, c_new)
            containing the kernels for the convolution
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b (numpy.ndarray): of shape (1, 1, 1, c_new)
            containing the biases applied to the convolution
        activation (function): is an activation function applied to
            the convolution
        padding (str): string that is either same or valid,
            indicating the type of padding used
        stride (tuple): (sh, sw) containing the strides
            sh is the stride for the height
            sw is the stride for the width

    Returns:
        numpy.ndarray: Output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    A_prev_padded = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    h_out = (h_prev + 2 * ph - kh) // sh + 1
    w_out = (w_prev + 2 * pw - kw) // sw + 1

    Z = np.zeros((m, h_out, w_out, c_new))

    for i in range(h_out):
        for j in range(w_out):
            vert_start = i * sh
            vert_end = vert_start + kh
            horiz_start = j * sw
            horiz_end = horiz_start + kw

            A_slice = A_prev_padded[:, vert_start:vert_end,
                                    horiz_start:horiz_end, :]
            for k in range(c_new):
                Z[:, i, j, k] = np.sum(A_slice * W[:, :, :, k],
                                       axis=(1, 2, 3)) + b[:, :, :, k]

    return activation(Z)
