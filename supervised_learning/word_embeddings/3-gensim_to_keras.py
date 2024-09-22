#!/usr/bin/env python3
"""
Extract & convert Word2Vec model to a keras Embedding layer
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a trained Gensim Word2Vec model to a Keras Embedding layer.

    Args:
        model (gensim.models.Word2Vec): A trained Word2Vec model from Gensim.

    Returns:
        keras.layers.Embedding: A Keras Embedding layer with the Word2Vec
        weights loaded, which can be further trained.
    """
    # Extract the word vectors from the trained Word2Vec model
    keyed_vectors = model.wv  # Holds the word vectors
    weights = keyed_vectors.vectors  # 2D numpy array of word vectors

    # Create a Keras Embedding layer using the Word2Vec weights
    layer = tf.keras.layers.Embedding(
        input_dim=weights.shape[0],  # Number of words (vocabulary size)
        output_dim=weights.shape[1],  # Dimensionality of the word vectors
        weights=[weights],  # Pre-trained weights from Word2Vec
        trainable=True  # Allow further training in Keras
    )
    return layer
