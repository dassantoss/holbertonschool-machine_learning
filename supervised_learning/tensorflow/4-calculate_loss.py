#!/usr/bin/env python3
"""
This Module defines a function to calculate the softmax cross-entropy
loss of predictions made by a neural network using TensorFlow.
"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction.
    Args:
        y (tf.placeholder): Placeholder for the labels of the
            input data.
        y_pred (tf.Tensor): Tensor containing the networks predictions.
            (logits)
    Returns:
        tf.Tensor: A tensor containing the loss of the prediction.
    """
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
