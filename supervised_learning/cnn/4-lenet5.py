#!/usr/bin/env python3
"""
LeNet-5 architecture in TensorFlow 1.x
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def lenet5(x, y):
    """
    Builds the LeNet-5 architecture using TensorFlow 1.x

    Args:
        x (tf.placeholder): shape (m, 28, 28, 1) containing input images
        y (tf.placeholder): shape (m, 10) containing one-hot labels

    Returns:
        y_pred (tf.Tensor): softmax activated output
        train_op (tf.Operation): training operation
        loss (tf.Tensor): loss of the network
        accuracy (tf.Tensor): accuracy of the network
    """
    he_init = tf.keras.initializers.VarianceScaling(scale=2.0)

    # Convolutional Layer 1
    C1 = tf.layers.conv2d(x, filters=6, kernel_size=5, padding='same',
                          kernel_initializer=he_init, activation=tf.nn.relu)

    # Pooling Layer 1
    S2 = tf.layers.max_pooling2d(C1, pool_size=2, strides=2)

    # Convolutional Layer 2
    C3 = tf.layers.conv2d(S2, filters=16, kernel_size=5, padding='valid',
                          kernel_initializer=he_init, activation=tf.nn.relu)

    # Pooling Layer 2
    S4 = tf.layers.max_pooling2d(C3, pool_size=2, strides=2)

    # Flatten layer
    S4_flat = tf.layers.flatten(S4)

    # Fully Connected Layer 1
    C5 = tf.layers.dense(S4_flat, units=120, kernel_initializer=he_init,
                         activation=tf.nn.relu)

    # Fully Connected Layer 2
    F6 = tf.layers.dense(C5, units=84, kernel_initializer=he_init,
                         activation=tf.nn.relu)

    # Output Layer
    y_pred = tf.layers.dense(F6, units=10, kernel_initializer=he_init,
                             activation=None)

    # Softmax output
    y_pred_softmax = tf.nn.softmax(y_pred)

    # Loss
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # Training Operation
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(y_pred_softmax, 1),
                                  tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return y_pred_softmax, train_op, loss, accuracy
