#!/usr/bin/env python3
"""
Transfer Knowledge
"""
from tensorflow import keras as K


def preprocess_data(X, Y):
    """
    Preprocess the data for the model
    """
    X_p = K.applications.densenet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


if __name__ == "__main__":
    # load data
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # preprocessing for DenseNet
    X_train, y_train = preprocess_data(x_train, y_train)
    X_test, y_test = preprocess_data(x_test, y_test)

    # Resize the input
    inputs = K.Input(shape=(32, 32, 3))
    resized_input = K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image, 7, 7, "channels_last")
    )(inputs)

    # create the base pre-trained model
    base_model = K.applications.DenseNet121(
        weights='imagenet', include_top=False, input_tensor=resized_input)
    base_model.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = K.layers.GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    x = K.layers.Dense(
        512, activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0),
        kernel_regularizer=K.regularizers.L2(0.01)
    )(x)
    x = K.layers.Dropout(0.5)(x)
    x = K.layers.Dense(
        256, activation='relu',
        kernel_initializer=K.initializers.HeNormal(seed=0),
        kernel_regularizer=K.regularizers.L2(0.01)
    )(x)
    x = K.layers.Dropout(0.5)(x)

    # logistic layer -- we have 10 classes
    predictions = K.layers.Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = K.Model(inputs=base_model.input, outputs=predictions)

    # compile (should be done *after* setting layers to non-trainable)
    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = K.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = K.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)

    # train the model on the new data for a few epochs
    model.fit(
        X_train, y_train, validation_data=(X_test, y_test),
        batch_size=128, epochs=50, verbose=1,
        callbacks=[early_stopping, reduce_lr])

    # save the model
    model.save('cifar10.h5')
