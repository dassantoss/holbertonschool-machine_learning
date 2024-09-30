#!/usr/bin/env python3
"""RNN Encoder for machine translation."""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNN Encoder class for encoding sequences."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNN Encoder.

        Args:
            vocab (int): Size of the input vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the GRU.
            batch (int): Batch size.
        """
        super(RNNEncoder, self).__init__()

        self.batch = batch
        self.units = units
        # Embedding layer to convert word indices to embedding vectors
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        # GRU layer for the RNN, initialized with glorot_uniform
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """Initialize the hidden state with zeros.

        Returns:
            Tensor: Tensor of shape (batch, units) filled with zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """Process the input sequence through the GRU.

        Args:
        x (Tensor): Input tensor of shape (batch, input_seq_len) containing
        word indices.
        initial (Tensor): Tensor of shape (batch, units) containing the
        initial hidden state.

        Returns:
        outputs (Tensor): Tensor of shape (batch, input_seq_len, units)
        containing the outputs.
        hidden (Tensor): Tensor of shape (batch, units) containing the last
        hidden state.
        """
        x = self.embedding(x)  # Convert word indices to embeddings
        outputs, hidden = self.gru(x, initial_state=initial)  # Pass GRU
        return outputs, hidden
