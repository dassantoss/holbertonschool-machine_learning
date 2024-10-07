#!/usr/bin/env python3
"""Train a Transformer model for machine translation"""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom learning rate schedule based on warmup steps"""

    def __init__(self, dm, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """Sparse categorical crossentropy loss function"""
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(
        real, pred, from_logits=True)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """Accuracy function for sparse categorical crossentropy"""
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Function to train the Transformer model for machine translation

    Args:
        N: Number of blocks in the encoder and decoder.
        dm: Dimensionality of the model.
        h: Number of heads in the multi-head attention.
        hidden: Number of hidden units in fully connected layers.
        max_len: Maximum number of tokens per sequence.
        batch_size: Batch size for training.
        epochs: Number of epochs to train for.

    Returns:
        The trained transformer model.
    """
    dataset = Dataset(batch_size, max_len)
    input_vocab_size = dataset.tokenizer_pt.vocab_size + 2
    target_vocab_size = dataset.tokenizer_en.vocab_size + 2

    transformer = Transformer(N, dm, h, hidden, input_vocab_size,
                              target_vocab_size, max_len, max_len)

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        print(f'Epoch {epoch+1} starting...')
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(dataset.data_train):
            with tf.GradientTape() as tape:
                # Create the correct masks for the encoder and decoder
                encoder_mask, combined_mask, decoder_mask = \
                    create_masks(inp, tar[:, :-1])

                predictions = transformer(inp, tar[:, :-1], True,
                                          encoder_mask, combined_mask,
                                          decoder_mask)
                loss = loss_function(tar[:, 1:], predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(zip(gradients,
                                          transformer.trainable_variables))

            train_loss(loss)
            acc = accuracy_function(tar[:, 1:], predictions)
            train_accuracy(acc)

            if batch % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch}: '
                      f'Loss {train_loss.result()}, '
                      f'Accuracy {train_accuracy.result()}')

        print(f'Epoch {epoch+1}: '
              f'Loss {train_loss.result()}, '
              f'Accuracy {train_accuracy.result()}')

    return transformer
