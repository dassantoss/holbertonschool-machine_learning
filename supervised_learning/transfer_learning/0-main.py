#!/usr/bin/env python3

from tensorflow import keras as K
import tensorflow as tf
preprocess_data = __import__('0-transfer').preprocess_data

# Custom object for Lambda layer
def resize_images(x):
    return tf.image.resize(x, (224, 224))

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)

# Load the model with custom_objects
model = K.models.load_model('cifar10.keras', custom_objects={'<lambda>': resize_images})
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)
