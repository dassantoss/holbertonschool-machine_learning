#!/usr/bin/env python3
"""
Sparse Autoencoder implementation with L1 regularization in TensorFlow/Keras
"""
import tensorflow.keras as keras


def build_encoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Builds the encoder part of the sparse autoencoder with L1 regularization.

    Parameters:
    - input_dims: integer, dimensions of the model input
    - hidden_layers: list of integers, number of nodes for each hidden layer
                     in the encoder
    - latent_dims: integer, dimensions of the latent space representation
    - lambtha: float, L1 regularization parameter for the latent space

    Returns:
    - keras.Model: The encoder model
    """
    encoder_input = keras.layers.Input(shape=(input_dims,))
    regularizer = keras.regularizers.L1(lambtha)

    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation='relu')(x)

    latent_space = keras.layers.Dense(latent_dims, activation='relu',
                                      activity_regularizer=regularizer)(x)

    return keras.Model(inputs=encoder_input, outputs=latent_space)


def build_decoder(latent_dims, hidden_layers, output_dims):
    """
    Builds the decoder part of the sparse autoencoder.

    Parameters:
    - latent_dims: integer, dimensions of the latent space representation
    - hidden_layers: list of integers, number of nodes for each hidden layer
                     in the encoder
    - output_dims: integer, dimensions of the model output

    Returns:
    - keras.Model: The decoder model
    """
    decoder_input = keras.layers.Input(shape=(latent_dims,))

    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation='relu')(x)

    decoder_output = keras.layers.Dense(output_dims, activation='sigmoid')(x)

    return keras.Model(inputs=decoder_input, outputs=decoder_output)


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Creates a sparse autoencoder with L1 regularization.

    Parameters:
    - input_dims: integer, dimensions of the model input
    - hidden_layers: list of integers, number of nodes for each hidden layer
                     in the encoder
    - latent_dims: integer, dimensions of the latent space representation
    - lambtha: float, L1 regularization parameter for the latent space

    Returns:
    - tuple: (encoder, decoder, auto), the models for the encoder, decoder,
              and full autoencoder
    """
    encoder = build_encoder(input_dims, hidden_layers, latent_dims, lambtha)
    decoder = build_decoder(latent_dims, hidden_layers, input_dims)

    encoder_input = keras.layers.Input(shape=(input_dims,))
    encoded_output = encoder(encoder_input)
    decoded_output = decoder(encoded_output)

    auto = keras.Model(inputs=encoder_input, outputs=decoded_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
