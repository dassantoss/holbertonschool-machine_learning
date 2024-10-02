#!/usr/bin/env python3
"""Scaled Dot Product Attention."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention.

    Args:
        Q (tensor): Query matrix of shape (..., seq_len_q, dk).
        K (tensor): Key matrix of shape (..., seq_len_v, dk).
        V (tensor): Value matrix of shape (..., seq_len_v, dv).
        mask (tensor, optional): Mask tensor that can be broadcasted into
            (..., seq_len_q, seq_len_v). Default is None.

    Returns:
        output (tensor): The result of applying the attention mechanism,
            shape (..., seq_len_q, dv).
        weights (tensor): The attention weights, shape (..., seq_len_q,
            seq_len_v).
    """
    # Step 1: Calculate the dot product between Q and K transpose
    # (..., seq_len_q, seq_len_v)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Step 2: Scale the result by the square root of the dimension of K
    dk = tf.cast(tf.shape(K)[-1], tf.float32)  # Dimension of K
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Step 3: If there is a mask, apply it by adding a very large negative
    #  number (-1e9) to the scaled logits where the mask is True
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Step 4: Apply the softmax to get the attention weights
    # (..., seq_len_q, seq_len_v)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # Step 5: Multiply the attention weights by the value matrix V
    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, dv)

    return output, attention_weights
