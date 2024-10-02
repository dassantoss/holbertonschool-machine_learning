#!/usr/bin/env python3
"""Transformer Encoder for machine translation."""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """Transformer Encoder class."""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the Encoder.

        Args:
            N (int): Number of encoder blocks.
            dm (int): Dimensionality of the model.
            h (int): Number of heads.
            hidden (int): Number of hidden units in the fully connected layer.
            input_vocab (int): Size of the input vocabulary.
            max_seq_len (int): Maximum sequence length possible.
            drop_rate (float): Dropout rate.
        """
        super(Encoder, self).__init__()
        self.dm = dm
        self.N = N
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = \
            [EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Forward pass through the encoder.

        Args:
            x (Tensor): Tensor of shape (batch, input_seq_len, dm), input to
            the encoder.
            training (bool): Whether the model is training.
            mask (Tensor): Mask to be applied for multi-head attention.

        Returns:
            Tensor: Tensor of shape (batch, input_seq_len, dm), encoder output.
        """
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)  # (batch, input_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x
