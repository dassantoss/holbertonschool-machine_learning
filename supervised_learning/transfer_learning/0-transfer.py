#!/usr/bin/env python3
"""
Transfer Knowledge with DenseNet121 on CIFAR-10
"""
from tensorflow import keras as K
import tensorflow as tf


def preprocess_data(X, Y):
    """
    Pre-processes the data for your model:

    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
        where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns: X_p, Y_p
        X_p is a numpy.ndarray containing the preprocessed X
        Y_p is a numpy.ndarray containing the preprocessed Y
    """
    # Preprocessing images using the DenseNet121 preprocessing function
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


def build_model():
    """
    Builds a DenseNet121 model with 10 classes, using ImageNet weights.
    The top layers are all frozen except the last few layers, to allow
    for fine-tuning on CIFAR-10.
    The first layer added is a lambda layer that scales up the data to
    the correct size for CIFAR-10.
    The following layers are added for feature classification.
    """
    # Load DenseNet121 model pre-trained on ImageNet
    base_model = K.applications.DenseNet121(weights='imagenet',
                                            include_top=False,
                                            input_shape=(224, 224, 3))

    # Freeze the base model
    base_model.trainable = False

    # Define the input layer
    inputs = K.Input(shape=(32, 32, 3))

    # Resize inputs to match the input size of DenseNet121
    resized_inputs = K.layers.Lambda(lambda x: tf.image.resize(x, (224, 224))
                                     )(inputs)

    # Pass the resized inputs through the base model
    base_model_output = base_model(resized_inputs, training=False)

    # Add classification layers on top of the base model
    x = K.layers.GlobalAveragePooling2D()(base_model_output)
    x = K.layers.Dropout(0.3)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    # Create the final model
    model = K.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == "__main__":
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Preprocess data
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)

    # Build and compile the model
    model = build_model()
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # Define callbacks
    early_stopping = K.callbacks.EarlyStopping(monitor='val_accuracy',
                                               patience=5,
                                               restore_best_weights=True)
    checkpoint = K.callbacks.ModelCheckpoint(filepath='best_model.keras',
                                             monitor='val_accuracy',
                                             save_best_only=True,
                                             mode='max')

    # Train the model
    history = model.fit(x_train, y_train,
                        epochs=10,
                        validation_data=(x_test, y_test),
                        callbacks=[early_stopping, checkpoint],
                        verbose=1)

    # Fine-tune the model
    model.trainable = True
    model.compile(optimizer=K.optimizers.Adam(1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    finetuning_history = model.fit(x_train, y_train,
                                   epochs=10,
                                   validation_data=(x_test, y_test),
                                   callbacks=[early_stopping, checkpoint],
                                   verbose=1)

    # Save the final model in h5 format
    model.save(filepath='cifar10.h5')

    # Evaluate on test set
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print(f'Test accuracy: {accuracy * 100:.2f}%')
