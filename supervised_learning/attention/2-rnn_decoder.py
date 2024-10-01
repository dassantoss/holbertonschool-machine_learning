#!/usr/bin/env python3
"""RNN Decoder for machine translation."""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNN Decoder class to decode for machine translation."""

    def __init__(self, vocab, embedding, units, batch):
        """Initialize the RNN Decoder.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vector.
            units (int): Number of hidden units in the RNN cell.
            batch (int): Batch size.
        """
        super(RNNDecoder, self).__init__()

        # Embedding layer to convert word indices to embedding vectors
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)

        # GRU layer for processing the decoder hidden states
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=False,
            return_state=True
        )

        # Attention mechanism
        self.attention = SelfAttention(units)

        # Dense layer to project the output of the GRU to the vocabulary
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """Perform a forward pass through the decoder.

        Args:
            x (Tensor): Shape (batch, 1), previous word in target sequence
            as an index.
            s_prev (Tensor): Shape (batch, units), previous decoder hidden
            state.
            hidden_states (Tensor): Shape (batch, input_seq_len, units),
            outputs of the encoder.

        Returns:
            y (Tensor): Shape (batch, vocab), output word as a one-hot vector.
            s (Tensor): Shape (batch, units), new decoder hidden state.
        """
        # Calculate the context vector using attention
        context, _ = self.attention(s_prev, hidden_states)

        # Convert the previous word (x) into an embedding
        x = self.embedding(x)

        # Remove the extra dimension -> (batch, embedding)
        x = tf.squeeze(x, axis=1)

        # Concatenate the context vector with the embedding previous word
        x_context = tf.concat([context, x], axis=-1)

        # Pass the concatenated vector to the GRU
        output, s = self.gru(
            tf.expand_dims(x_context, axis=1),
            initial_state=s_prev
        )

        # Project the output of the GRU to the vocabulary space
        y = self.F(output)

        return y, s
