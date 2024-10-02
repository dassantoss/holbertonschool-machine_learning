#!/usr/bin/env python3
"""Transformer Decoder Block for machine translation."""

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """DecoderBlock class to create a block for the Transformer decoder."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the DecoderBlock.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the fully connected layer.
            drop_rate (float): Dropout rate, default is 0.1.
        """
        super(DecoderBlock, self).__init__()
        # First Multi-Head Attention layer (masked for look-ahead)
        self.mha1 = MultiHeadAttention(dm, h)
        # Second Multi-Head Attention layer
        # (regular attention with encoder output)
        self.mha2 = MultiHeadAttention(dm, h)
        # Hidden dense layer with ReLU activation
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # Output dense layer
        self.dense_output = tf.keras.layers.Dense(dm)
        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """Forward pass through the decoder block.

        Args:
            x (tensor): Input tensor of shape (batch, target_seq_len, dm).
            encoder_output (tensor): Output from the encoder of shape
                                     (batch, input_seq_len, dm).
            training (bool): Whether the model is in training mode.
            look_ahead_mask (tensor or None): Mask to apply for look-ahead
                                              in the first attention layer.
            padding_mask (tensor or None): Mask to apply for padding in the
                                           second attention layer.

        Returns:
            tensor: Output tensor of shape (batch, target_seq_len, dm).
        """
        # First multi-head attention (self-attention with look-ahead mask)
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        # Second multi-head attention (with encoder output and padding mask)
        attn2, _ = self.mha2(out1, encoder_output, encoder_output,
                             padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        # Feed-forward network
        ffn_output = self.dense_hidden(out2)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3
