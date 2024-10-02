#!/usr/bin/env python3
"""Transformer Decoder for machine translation."""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """Transformer Decoder class."""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Initialize the Decoder.

        Args:
            N (int): Number of decoder blocks.
            dm (int): Dimensionality of the model.
            h (int): Number of heads.
            hidden (int): Number of hidden units in the fully connected layer.
            target_vocab (int): Size of the target vocabulary.
            max_seq_len (int): Maximum sequence length possible.
            drop_rate (float): Dropout rate.
        """
        super(Decoder, self).__init__()
        self.dm = dm
        self.N = N
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = \
            [DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Forward pass through the decoder.

        Args:
            x (Tensor): Tensor of shape (batch, target_seq_len, dm), input to
            the decoder.
            encoder_output (Tensor): Tensor of shape (batch, input_seq_len,
            dm), output from the encoder.
            training (bool): Whether the model is training.
            look_ahead_mask (Tensor): Mask for the first multi-head attention.
            padding_mask (Tensor): Mask for the second multi-head attention.

        Returns:
            Tensor: Tensor of shape (batch, target_seq_len, dm), decoder
            output.
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        # Add embedding and positional encoding
        x = self.embedding(x)  # (batch, target_seq_len, dm)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len, :]

        # Apply dropout
        x = self.dropout(x, training=training)

        # Pass through each decoder block
        for block in self.blocks:
            x = block(x, encoder_output, training, look_ahead_mask,
                      padding_mask)

        return x
