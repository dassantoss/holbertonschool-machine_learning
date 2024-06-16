#!/usr/bin/env python3
"""DenseNet-121 Implementation"""
from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture as described in Densely
    Connected Convolutional Networks.

    Args:
        growth_rate (int): growth rate.
        compression (float): compression factor.

    Returns:
        keras.Model: Keras model of the DenseNet-121 architecture.
    """
    inputs = K.Input(shape=(224, 224, 3))

    # Initial Convolution and Pooling
    x = K.layers.BatchNormalization()(inputs)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                        kernel_initializer=K.initializers.he_normal(seed=0))(x)
    x = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Dense Block 1
    x, nb_filters = dense_block(x, 64, growth_rate, 6)

    # Transition Layer 1
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 2
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 12)

    # Transition Layer 2
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 3
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 24)

    # Transition Layer 3
    x, nb_filters = transition_layer(x, nb_filters, compression)

    # Dense Block 4
    x, nb_filters = dense_block(x, nb_filters, growth_rate, 16)

    # Final Batch Normalization
    x = K.layers.BatchNormalization()(x)

    # Global Average Pooling
    x = K.layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer
    outputs = \
        K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=K.initializers.he_normal(seed=0)
                       )(x)

    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
