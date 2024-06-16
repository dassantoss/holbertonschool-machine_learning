#!/usr/bin/env python3
"""Transition Layer Implementation"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Densely
    Connected Convolutional Networks.

    Args:
        X (tensor): output from the previous layer.
        nb_filters (int): number of filters in X.
        compression (float): compression factor for the transition layer.

    Returns:
        tensor: output of the transition layer.
        int: number of filters within the output.
    """
    # Batch Normalization and ReLU activation
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    # 1x1 Convolution with compression factor
    nb_filters = int(nb_filters * compression)
    X = K.layers.Conv2D(nb_filters, (1, 1), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0)
                        )(X)

    # Average Pooling
    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2), padding='same')(X)

    return X, nb_filters
