#!/usr/bin/env python3
"""Dense Block Implementation"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Densely Connected
    Convolutional Networks.

    Args:
        X (tensor): output from the previous layer.
        nb_filters (int): number of filters in X.
        growth_rate (int): growth rate for the dense block.
        layers (int): number of layers in the dense block.

    Returns:
        tensor: concatenated output of each layer within the dense block.
        int: number of filters within the concatenated outputs.
    """
    for i in range(layers):
        # Bottleneck layer
        bn = K.layers.BatchNormalization()(X)
        relu = K.layers.Activation('relu')(bn)
        bottleneck = \
            K.layers.Conv2D(4 * growth_rate, (1, 1), padding='same',
                            kernel_initializer=K.initializers.he_normal(seed=0)
                            )(relu)

        # Composite function (3x3 convolution)
        bn = K.layers.BatchNormalization()(bottleneck)
        relu = K.layers.Activation('relu')(bn)
        conv = \
            K.layers.Conv2D(growth_rate, (3, 3), padding='same',
                            kernel_initializer=K.initializers.he_normal(seed=0)
                            )(relu)

        # Concatenate the input with the output
        X = K.layers.Concatenate()([X, conv])
        nb_filters += growth_rate

    return X, nb_filters
