#!/usr/bin/env python3
"""
This Module defines a function to calculate the accuracy of
predictions made by a neural network using TensorFlow.
"""
import tensorflow.compat.v1 as tf  # type: ignore
tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.
    Args:
        y (tf.placeholder): Placeholder for the labels of the
            input data.
        y_pred (tf.Tensor): Tensor containing the networks predictions.
    Returns:
        tf.Tensor: A tensor containing the decimal accuracy of the
            prediction.
    """
    prediction = tf.argmax(y_pred, 1)
    correct_label = tf.argmax(y, 1)
    correct_predictions = tf.equal(prediction, correct_label)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
