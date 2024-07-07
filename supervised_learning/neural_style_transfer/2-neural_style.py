#!/usr/bin/env python3
"""
Neural Style Transfer
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    NST class performs Neural Style Transfer.

    Attributes:
        style_layers (list): List of layers to be used for style extraction.
        content_layer (str): Layer to be used for content extraction.
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class with style and content images and their
        weights.

        Args:
            style_image (numpy.ndarray): The image used as a style reference.
            content_image (numpy.ndarray): The image used as a content
            reference.
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.

        Raises:
            TypeError: If style_image or content_image is not a numpy.ndarray
            with shape (h, w, 3).
            TypeError: If alpha or beta is not a non-negative number.
        """
        # Validate style_image
        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        # Validate content_image
        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        # Validate alpha
        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        # Validate beta
        if not isinstance(beta, (float, int)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Scale and preprocess style and content images
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        # Load the model
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Scales an image so that its pixels values are between 0 and 1 and
        its largest side is 512 pixels.

        Args:
            image (numpy.ndarray): The image to be scaled.

        Raises:
            TypeError: If image is not a numpy.ndarray with shape (h, w, 3).

        Returns:
            tf.Tensor: The scaled image with shape (1, h_new, w_new, 3) where
            max(h_new, w_new) == 512.
        """
        # Validate image
        if (not isinstance(image, np.ndarray) or image.shape[-1] != 3):
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        # Get original dimensions
        h, w = image.shape[:2]

        # Calculate scale factor to resize the largest side to 512 pixels
        if w > h:
            new_w = 512
            new_h = int((h * 512) / w)
        else:
            new_h = 512
            new_w = int((w * 512) / h)

        # Resize image (with bicubic interpolation)
        image_resized = tf.image.resize(
            image, size=[new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC)

        # Normalize pixel values to the range [0, 1]
        image_normalized = image_resized / 255

        # Clip values to ensure they are within [0, 1] range
        image_clipped = tf.clip_by_value(image_normalized, 0, 1)

        # Add batch dimension on axis 0 and return
        return tf.expand_dims(image_clipped, axis=0)

    def load_model(self):
        """
        Creates the model used to calculate cost.

        The model uses the VGG19 Keras model as a base. The input of the model
        is the same as the VGG19 input, and the output is a list containing
        the outputs of the VGG19 layers listed in style_layers followed by
        content_layer.

        Returns:
            model (tf.keras.Model): The Keras model used to calculate cost.
        """
        # Load VGG19 model from Keras API
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        vgg.trainable = False
        # Replace MaxPooling2D layers with AveragePooling2D layers
        for layer in vgg.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

        # get outputs of the style and content layers from modified VGG19
        style_outputs = [vgg.get_layer(
            name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        # Create the model, make it non-trainable and return it
        self.model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=style_outputs + [content_output])

    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of an input layer.

        Args:
            input_layer (tf.Tensor or tf.Variable): A tensor of shape
            (1, h, w, c) containing the layer output whose gram matrix
            should be calculated.

        Raises:
            TypeError: If input_layer is not a tensor of rank 4.

        Returns:
            tf.Tensor: A tensor of shape (1, c, c) containing the gram matrix
            of input_layer.
        """
        # Calidate input_layer rank and batch size
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable))
                or len(input_layer.shape) != 4
                or input_layer.shape[0] != 1):
            raise TypeError("input_layer must be a tensor of rank 4")

        # calculate gram matrix: (batch, height, width, channel)
        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)

        # Normalize by number of locations (h * w) then return gram tensor
        input_shape = tf.shape(input_layer)
        nb_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return gram / nb_locations
