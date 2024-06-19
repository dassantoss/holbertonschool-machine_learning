#!/usr/bin/env python3

from tensorflow import keras
import tensorflow as tf
preprocess_data = __import__('0-transfer').preprocess_data

_, (X, Y) = keras.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = keras.models.load_model('cifar10.h5', custom_objects={'tf': tf})
model.evaluate(X_p, Y_p, batch_size=128, verbose=1)