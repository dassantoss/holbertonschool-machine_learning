#!/usr/bin/env python3
"""Positional Encoding for Transformer."""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer.

    Args:
        max_seq_len (int): Maximum sequence length.
        dm (int): Model depth (dimensionality).

    Returns:
        np.ndarray: Positional encoding of shape (max_seq_len, dm).
    """
    # Initialize the positional encoding matrix with zeros
    PE = np.zeros((max_seq_len, dm))

    # Create a matrix with positions (0 to max_seq_len-1)
    positions = np.arange(max_seq_len)[:, np.newaxis]

    # Create a matrix with dimensions (0 to dm-1) and apply the scaling
    dimensions = np.arange(dm)[np.newaxis, :]

    # Compute the angles for the positional encoding using the formula
    angle_rates = 1 / np.power(10000, (dimensions // 2 * 2) / np.float32(dm))

    # Apply sine to even indices (2i) and cosine to odd indices (2i+1)
    PE[:, 0::2] = np.sin(positions * angle_rates[:, 0::2])
    PE[:, 1::2] = np.cos(positions * angle_rates[:, 1::2])

    return PE
