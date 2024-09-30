#!/usr/bin/env python3
"""Self-Attention layer for machine translation."""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Self-Attention class to compute attention for machine translation."""

    def __init__(self, units):
        """Initialize the SelfAttention layer.

        Args:
        units (int): Number of hidden units in the alignment model.
        """
        super(SelfAttention, self).__init__()

        # Dense layer to apply to the previous decoder hidden state (W)
        self.W = tf.keras.layers.Dense(units)
        # Dense layer to apply to the encoder hidden states (U)
        self.U = tf.keras.layers.Dense(units)
        # Dense layer to apply to the tanh of the sum of W(s_prev)
        # and U(hidden_states)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """Calculate the attention weights and context vector.

        Args:
        s_prev (Tensor): Shape (batch, units) containing the previous
        decoder hidden state.
        hidden_states (Tensor): Shape (batch, input_seq_len, units)
        containing the encoder hidden states.

        Returns:
        context (Tensor): Shape (batch, units), the context vector for the
        decoder.
        weights (Tensor): Shape (batch, input_seq_len, 1), the attention
        weights.
        """
        # Expand the previous hidden state (s_prev) to match the encoder
        # hidden states dimensions
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)  # (batch, 1, units)

        # Apply the dense layers to s_prev and hidden_states
        W_s = self.W(s_prev_expanded)  # (batch, 1, units)
        U_h = self.U(hidden_states)    # (batch, input_seq_len, units)

        # Compute alignment scores
        score = tf.nn.tanh(W_s + U_h)  # (batch, input_seq_len, units)
        # Apply final dense layer (V) to compute unnormalized attention
        # scores
        attention_scores = self.V(score)  # (batch, input_seq_len, 1)

        # Apply softmax to obtain the attention weights
        # (batch,input_seq_len, 1)
        weights = tf.nn.softmax(attention_scores, axis=1)

        # Compute the context vector as the weighted sum of the encoder
        # hidden states
        # (batch, units)
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
