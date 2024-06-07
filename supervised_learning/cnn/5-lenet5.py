#!/usr/bin/env python3
"""
LeNet-5 architecture in Keras
"""
from tensorflow import keras as K


def lenet5(X):
    """
    Builds the LeNet-5 architecture using Keras

    Args:
        X (K.Input): shape (m, 28, 28, 1) containing the input images

    Returns:
        model (K.Model): compiled Keras model
    """
    he_init = K.initializers.VarianceScaling(scale=2.0, seed=0)

    # Convolutional Layer 1
    C1 = K.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same',
                         kernel_initializer=he_init, activation='relu')(X)

    # Pooling Layer 1
    S2 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C1)

    # Convolutional Layer 2
    C3 = K.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid',
                         kernel_initializer=he_init, activation='relu')(S2)

    # Pooling Layer 2
    S4 = K.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(C3)

    # Flatten layer
    S4_flat = K.layers.Flatten()(S4)

    # Fully Connected Layer 1
    C5 = K.layers.Dense(units=120, kernel_initializer=he_init,
                        activation='relu')(S4_flat)

    # Fully Connected Layer 2
    F6 = K.layers.Dense(units=84, kernel_initializer=he_init,
                        activation='relu')(C5)

    # Output Layer
    y_pred = K.layers.Dense(units=10, kernel_initializer=he_init,
                            activation='softmax')(F6)

    # Model
    model = K.Model(inputs=X, outputs=y_pred)

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
