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
    # Setup stride, matrixes and kernel dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        ph = 0
        pw = 0

    # Calculate output dimensions
    output_h = (h_prev + 2 * ph - kh) // sh + 1
    output_w = (w_prev + 2 * pw - kw) // sw + 1

    # Padding input as needed
    padded_A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                           mode='constant')

    # Initialize convolution output array
    convolved = np.zeros((m, output_h, output_w, c_new))

    for i in range(output_h):
        for j in range(output_w):
            # Extract region from padded input
            region = padded_A_prev[:, i*sh:i*sh+kh, j*sw:j*sw+kw, :]
            for k in range(c_new):
                # Convolve each input (m) in the region, using kernel k
                convolved[:, i, j, k] = np.sum((region * W[:, :, :, k]),
                                               axis=(1, 2, 3))

    # Layer l activation output: A(l+1) = g(Z), with Z = A(l) * W + b
    return activation(convolved + b)
