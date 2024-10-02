#!/usr/bin/env python3
"""MultiHead Attention for machine translation."""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """MultiHeadAttention class to perform multi-head attention."""

    def __init__(self, dm, h):
        """Initialize the MultiHeadAttention layer.

        Args:
            dm (int): The dimensionality of the model.
            h (int): The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()

        if dm % h != 0:
            raise ValueError("dm must be divisible by h")

        self.h = h
        self.dm = dm
        self.depth = dm // h

        # Layers to generate Query, Key, and Value matrices
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        # Linear layer to combine the outputs of all attention heads
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth) and transpose.

        Args:
            x (tensor): Input tensor to be split.
            batch_size (int): The batch size.

        Returns:
            Tensor: Transposed tensor of shape (batch_size, h, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Perform multi-head attention.

        Args:
            Q (tensor): Query matrix.
            K (tensor): Key matrix.
            V (tensor): Value matrix.
            mask (tensor or None): Mask to apply.

        Returns:
            output (tensor): Multi-head attention output.
            weights (tensor): Attention weights.
        """
        batch_size = tf.shape(Q)[0]

        # Generate Q, K, V matrices
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Split Q, K, V into multiple heads
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Apply scaled dot-product attention
        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        # Concatenate the attention output for all heads
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.dm))

        # Apply the final linear layer
        output = self.linear(concat_attention)

        return output, weights
