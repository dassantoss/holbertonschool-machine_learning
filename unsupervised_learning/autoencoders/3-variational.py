#!/usr/bin/env python3
"""
Variational Autoencoder implementation in TensorFlow/Keras
"""
import tensorflow.keras as keras


def sampling(args, latent_dims):
    """
    Samples from a distribution (reparameterization trick).

    Parameters:
    - args: tuple containing the mean and log variance of the latent space
    - latent_dims: integer, dimensions of the latent space representation

    Returns:
    - z: the sampled latent vector
    """
    z_mean, z_log_var = args
    epsilon = \
        keras.backend.random_normal(shape=(keras.backend.shape(z_mean)[0],
                                           latent_dims))
    return z_mean + keras.backend.exp(z_log_var / 2) * epsilon


def build_encoder(input_dims, hidden_layers, latent_dims):
    """
    Builds the encoder part of the variational autoencoder.

    Parameters:
    - input_dims: integer, dimensions of the model input
    - hidden_layers: list of integers, number of nodes for each hidden layer
    in the encoder
    - latent_dims: integer, dimensions of the latent space representation

    Returns:
    - model_encoder: the encoder model
    - mean: mean vector of the latent distribution
    - log_var: log variance vector of the latent distribution
    """
    encoder_input = keras.layers.Input(shape=(input_dims,))

    x = encoder_input
    for units in hidden_layers:
        x = keras.layers.Dense(units=units, activation='relu')(x)

    mean = keras.layers.Dense(latent_dims, activation=None)(x)
    log_var = keras.layers.Dense(latent_dims, activation=None)(x)

    z = keras.layers.Lambda(sampling, output_shape=(latent_dims,),
                            arguments={'latent_dims': latent_dims})([mean,
                                                                     log_var])

    model_encoder = keras.Model(inputs=encoder_input, outputs=[z, mean,
                                                               log_var])
    return model_encoder, mean, log_var


def build_decoder(latent_dims, hidden_layers, output_dims):
    """
    Builds the decoder part of the variational autoencoder.

    Parameters:
    - latent_dims: integer, dimensions of the latent space representation
    - hidden_layers: list of integers, number of nodes for each hidden
                     layer in the decoder
    - output_dims: integer, dimensions of the model output

    Returns:
    - model_decoder: the decoder model
    """
    decoder_input = keras.layers.Input(shape=(latent_dims,))

    x = decoder_input
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units=units, activation='relu')(x)

    decoder_output = keras.layers.Dense(output_dims, activation='sigmoid')(x)

    model_decoder = keras.models.Model(inputs=decoder_input,
                                       outputs=decoder_output)
    return model_decoder


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Creates the full variational autoencoder model.

    Parameters:
    - input_dims: integer, dimensions of the model input
    - hidden_layers: list of integers, number of nodes for each hidden layer
                     in the encoder
    - latent_dims: integer, dimensions of the latent space representation

    Returns:
    - encoder: the encoder model
    - decoder: the decoder model
    - auto: the full variational autoencoder model
    """
    encoder, mean, log_var = build_encoder(input_dims, hidden_layers,
                                           latent_dims)
    decoder = build_decoder(latent_dims, hidden_layers, input_dims)

    encoder_input = encoder.input
    z, mean, log_var = encoder(encoder_input)
    decoded_output = decoder(z)

    auto = keras.Model(inputs=encoder_input, outputs=decoded_output)

    reconstruction_loss = keras.losses.binary_crossentropy(encoder_input,
                                                           decoded_output)
    reconstruction_loss *= input_dims

    kl_loss = \
        1 + log_var - keras.backend.square(mean) - keras.backend.exp(log_var)
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)
    auto.add_loss(vae_loss)
    auto.compile(optimizer='adam')

    return encoder, decoder, auto
