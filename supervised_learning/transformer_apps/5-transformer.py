#!/usr/bin/env python3
"""Full Transformer Network"""
import tensorflow as tf
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


class Transformer(tf.keras.Model):
    """Transformer class for machine translation."""

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """Initialize the Transformer.

        Args:
            N (int): Number of encoder and decoder blocks.
            dm (int): Dimensionality of the model.
            h (int): Number of heads.
            hidden (int): Number of hidden units in the fully connected
            layers.
            input_vocab (int): Size of the input vocabulary.
            target_vocab (int): Size of the target vocabulary.
            max_seq_input (int): Maximum sequence length for input.
            max_seq_target (int): Maximum sequence length for target.
            drop_rate (float): Dropout rate.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_seq_input,
                               drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_seq_target,
                               drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """Forward pass through the Transformer.

        Args:
            inputs (Tensor): Tensor of shape (batch, input_seq_len),
            input sequences.
            target (Tensor): Tensor of shape (batch, target_seq_len),
            target sequences.
            training (bool): Whether the model is training.
            encoder_mask (Tensor): Padding mask for the encoder.
            look_ahead_mask (Tensor): Look-ahead mask for the decoder.
            decoder_mask (Tensor): Padding mask for the decoder.

        Returns:
            Tensor: Output of the transformer of shape (batch, target_seq_len,
            target_vocab).
        """
        # Pass inputs through the encoder
        # (batch, input_seq_len, dm)
        enc_output = self.encoder(inputs, training, encoder_mask)

        # Pass target through the decoder
        # (batch, target_seq_len, dm)
        dec_output = self.decoder(target, enc_output, training,
                                  look_ahead_mask, decoder_mask)

        # Final dense layer to map to target vocabulary size
        # (batch, target_seq_len, target_vocab)
        final_output = self.linear(dec_output)

        return final_output
