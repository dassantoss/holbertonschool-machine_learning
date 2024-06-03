#!/usr/bin/env python3
"""
Trains a model using mini-batch gradient descent, analyzes validation data,
applies early stopping, uses learning rate decay, and saves the best model.
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True,
                shuffle=False):
    """
    Trains a model using mini-batch gradient descent and analyzes validation
    data.

    Args:
        network (keras.Model): The model to train.
        data (numpy.ndarray): Input data of shape (m, nx).
        labels (numpy.ndarray): One-hot labels of shape (m, classes).
        batch_size (int): Size of the batch used for mini-batch gradient
            descent.
        epochs (int): Number of passes through data for mini-batch gradient
            descent.
        validation_data (tuple): Data to validate the model with, if not None.
        early_stopping (bool): Indicates whether early stopping should be used.
        patience (int): Patience used for early stopping.
        learning_rate_decay (bool): Indicates whether learning rate decay
            should be used.
        alpha (float): Initial learning rate.
        decay_rate (float): Decay rate for learning rate decay.
        save_best (bool): Indicates whether to save the best model.
        filepath (str): File path where the model should be saved.
        verbose (bool): Determines if output should be printed during training.
        shuffle (bool): Determines whether to shuffle the batches every epoch.

    Returns:
        keras.callbacks.History: The History object generated after training
        the model.
    """
    callbacks = []
    if early_stopping and validation_data:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True
        )
        callbacks.append(early_stop)

    if learning_rate_decay and validation_data:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(
            scheduler,
            verbose=1
        )
        callbacks.append(lr_decay)

    if save_best and validation_data:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(checkpoint)

    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
    return history
