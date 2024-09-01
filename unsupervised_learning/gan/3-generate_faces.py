#!/usr/bin/env python3
"""
GAN Generator and Discriminator for face generation.
"""
import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds the generator and discriminator models using convolutional layers.

    Returns:
        generator (tf.keras.Model): The generator model.
        discriminator (tf.keras.Model): The discriminator model.
    """

    def generator():
        """
        Builds the generator model.
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(16,)),
            keras.layers.Dense(2048),
            keras.layers.Reshape((2, 2, 512)),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2D(64, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('tanh'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2D(16, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('tanh'),
            keras.layers.UpSampling2D(),
            keras.layers.Conv2D(1, (3, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Activation('tanh')
        ], name='generator')
        return model

    def discriminator():
        """
        Builds the discriminator model.
        """
        model = keras.Sequential([
            keras.layers.Input(shape=(16, 16, 1)),
            keras.layers.Conv2D(32, (3, 3), padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Activation('tanh'),
            keras.layers.Conv2D(64, (3, 3), padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Activation('tanh'),
            keras.layers.Conv2D(128, (3, 3), padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Activation('tanh'),
            keras.layers.Conv2D(256, (3, 3), padding='same'),
            keras.layers.MaxPooling2D(),
            keras.layers.Activation('tanh'),
            keras.layers.Flatten(),
            keras.layers.Dense(1)
        ], name='discriminator')
        return model

    return generator(), discriminator()
