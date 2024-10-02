#!/usr/bin/env python3
"""Complete Transformer Network."""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


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
