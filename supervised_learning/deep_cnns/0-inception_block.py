#!/usr/bin/env python3
"""Inception Block Implementation"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Function thats Builds an inception block as described in Going Deeper
    with Convolutions (2014).

    Args:
        A_prev (tensor): output from the previous layer.
        filters (list/tuple): tuple or list containing F1,F3R,F3,F5R,F5,FPP.

    Returns:
        tensor: concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 Convolution
    conv_1x1 = K.layers.Conv2D(filters=F1, kernel_size=(1, 1),
                               padding='same', activation='relu')(A_prev)

    # 1x1 Convolution followed by 3x3 Convolution
    conv_3x3_reduce = K.layers.Conv2D(filters=F3R, kernel_size=(1, 1),
                                      padding='same',
                                      activation='relu')(A_prev)
    conv_3x3 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3),
                               padding='same',
                               activation='relu')(conv_3x3_reduce)

    # 1x1 Convolution followed by 5x5 Convolution
    conv_5x5_reduce = K.layers.Conv2D(filters=F5R, kernel_size=(1, 1),
                                      padding='same',
                                      activation='relu')(A_prev)
    conv_5x5 = K.layers.Conv2D(filters=F5, kernel_size=(5, 5),
                               padding='same',
                               activation='relu')(conv_5x5_reduce)

    # Max Pooling followed by 1x1 Convolution
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1),
                                     padding='same')(A_prev)
    conv_pool_proj = K.layers.Conv2D(filters=FPP, kernel_size=(1, 1),
                                     padding='same',
                                     activation='relu')(max_pool)

    # Concatenate all the filters
    output = K.layers.Concatenate(axis=-1)([conv_1x1, conv_3x3,
                                            conv_5x5, conv_pool_proj])

    return output
