#!/usr/bin/env python3
"""Transformer Encoder Block for machine translation."""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """EncoderBlock class to create a block for the Transformer encoder."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the EncoderBlock.

        Args:
            dm (int): Dimensionality of the model.
            h (int): Number of attention heads.
            hidden (int): Number of hidden units in the fully connected layer.
            drop_rate (float): Dropout rate, default is 0.1.
        """
        super(EncoderBlock, self).__init__()
        # Multi-Head Attention layer
        self.mha = MultiHeadAttention(dm, h)
        # Hidden dense layer
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        # Output dense layer
        self.dense_output = tf.keras.layers.Dense(dm)
        # Layer normalization layers
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Forward pass through the encoder block.

        Args:
            x (tensor): Input tensor of shape (batch, input_seq_len, dm).
            training (bool): Whether the model is in training mode.
            mask (tensor or None): Mask to apply for attention.

        Returns:
            tensor: Output tensor of shape (batch, input_seq_len, dm).
        """
        # Multi-head attention
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed forward neural network
        ffn_output = self.dense_hidden(out1)
        ffn_output = self.dense_output(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2
