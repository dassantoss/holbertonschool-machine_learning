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
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

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

        # Pass the previous word index through the embedding layer
        x = self.embedding(x)

        # Concatenate the context vector with x (expand context dimension)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        # Pass the concatenated vector through the GRU layer
        output, s = self.gru(x)

        # Remove the extra axis
        output = tf.squeeze(output, axis=1)

        # Pass the GRU output through the Dense layer to predict the next word
        y = self.F(output)

        return y, s
