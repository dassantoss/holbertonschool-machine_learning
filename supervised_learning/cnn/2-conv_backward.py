#!/usr/bin/env python3
"""
Performs back propagation over a convolutional layer
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Function that performs back propagation over a convolutional layer

    Args:
        dZ (numpy.ndarray): of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output
            of the convolutional layer
        A_prev (numpy.ndarray): of shape (m, h_prev, w_prev, c_prev)
            containing the output of the previous layer
        W (numpy.ndarray): of shape (kh, kw, c_prev, c_new) containing
            the kernels for the convolution
        b (numpy.ndarray): of shape (1, 1, 1, c_new) containing the biases
            applied to the convolution
        padding (str): string that is either 'same' or 'valid',
            indicating the type of padding used
        stride (tuple): (sh, sw) containing the strides
            sh is the stride for the height
            sw is the stride for the width

    Returns:
        dA_prev (numpy.ndarray): Partial derivatives with respect to the
            previous layer
        dW (numpy.ndarray): Partial derivatives with respect to the kernels
        db (numpy.ndarray): Partial derivatives with respect to the biases
    """
    # Setup stride, padding, and shapes
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride
    m, h_new, w_new, _ = dZ.shape

    if padding == "same":
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        ph, pw = 0, 0

    A_prev_padded = np.pad(
        A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), mode='constant')
    dA_prev_padded = np.pad(
        np.zeros_like(A_prev), ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant')
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Iterate over the gradients
    for i in range(h_new):
        for j in range(w_new):
            for k in range(c_new):
                vert_start = i * sh
                vert_end = vert_start + kh
                horiz_start = j * sw
                horiz_end = horiz_start + kw

                A_slice = A_prev_padded[
                    :, vert_start:vert_end, horiz_start:horiz_end, :]
                dA_prev_padded[
                    :, vert_start:vert_end, horiz_start:horiz_end, :] += (
                    W[:, :, :, k] * dZ[:, i, j, k][:, np.newaxis,
                                                   np.newaxis, np.newaxis])
                dW[:, :, :, k] += np.sum(
                    A_slice * dZ[:, i, j, k][:, np.newaxis, np.newaxis,
                                             np.newaxis], axis=0)

    if padding == 'same':
        dA_prev = dA_prev_padded[:, ph:-ph, pw:-pw, :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db
