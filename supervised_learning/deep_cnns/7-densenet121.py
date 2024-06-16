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
    # Initialize he_normal with seed 0
    init = K.initializers.HeNormal(seed=0)

    # Input tensor (assuming given shape of data)
    inputs = K.Input(shape=(224, 224, 3))

    # Initial convolution and pooling
    # Number of filters set to twice the growth rate
    nb_filters = growth_rate * 2

    # Batch normalization
    norm = K.layers.BatchNormalization()(inputs)

    # ReLU activation
    activ = K.layers.Activation(activation="relu")(norm)

    # Convolution with initial number of filters
    conv = K.layers.Conv2D(filters=nb_filters,
                           kernel_size=(7, 7),
                           strides=(2, 2),
                           padding="same",
                           kernel_initializer=init)(activ)

    # Max pooling layer
    max_pool = K.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding="same")(conv)

    # First dense block and transition layer
    block1, nb_filters = dense_block(max_pool, nb_filters, growth_rate, 6)
    trans1, nb_filters = transition_layer(block1, nb_filters, compression)

    # Second dense block and transition layer
    block2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)
    trans2, nb_filters = transition_layer(block2, nb_filters, compression)

    # Third dense block and transition layer
    block3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)
    trans3, nb_filters = transition_layer(block3, nb_filters, compression)

    # Fourth dense block
    block4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    # Final batch normalization and activation
    norm_final = K.layers.BatchNormalization()(block4)
    activ_final = K.layers.Activation('relu')(norm_final)

    # Global average pooling
    avg_pool = K.layers.GlobalAveragePooling2D()(activ_final)

    # Fully connected layer with softmax activation
    outputs = K.layers.Dense(1000, activation='softmax',
                             kernel_initializer=init)(avg_pool)

    # Create Keras model
    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
