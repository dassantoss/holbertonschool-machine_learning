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
        style_layers (list): Layers for style extraction.
        content_layer (str): Layer for content extraction.
    """
    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize NST class with style and content images.

        Args:
            style_image (numpy.ndarray): Style reference image.
            content_image (numpy.ndarray): Content reference image.
            alpha (float): Weight for content cost.
            beta (float): Weight for style cost.

        Raises:
            TypeError: If inputs are not valid.
        """
        if not isinstance(style_image, np.ndarray) or \
                style_image.shape[-1] != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(content_image, np.ndarray) or \
                content_image.shape[-1] != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)"
            )
        if not isinstance(alpha, (float, int)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (float, int)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescale image to max 512px and normalize.

        Args:
            image (numpy.ndarray): Image to be scaled.

        Raises:
            TypeError: If image is not valid.

        Returns:
            tf.Tensor: Scaled image.
        """
        if not isinstance(image, np.ndarray) or \
                image.shape[-1] != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)"
            )

        h, w = image.shape[:2]
        if w > h:
            new_w = 512
            new_h = int((h * 512) / w)
        else:
            new_h = 512
            new_w = int((w * 512) / h)

        image_resized = tf.image.resize(
            image, size=[new_h, new_w], method=tf.image.ResizeMethod.BICUBIC
        )
        image_normalized = image_resized / 255
        image_clipped = tf.clip_by_value(image_normalized, 0, 1)
        return tf.expand_dims(image_clipped, axis=0)

    def load_model(self):
        """
        Load VGG19 model with average pooling.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet'
        )
        vgg.trainable = False

        selected_outputs = [
            vgg.get_layer(name).output for name in (
                self.style_layers + [self.content_layer]
            )
        ]
        self.model = tf.keras.models.Model(
            inputs=vgg.input, outputs=selected_outputs
        )

        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                layer.__class__ = tf.keras.layers.AveragePooling2D

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculate gram matrix.

        Args:
            input_layer (tf.Tensor): Layer output tensor.

        Raises:
            TypeError: If input_layer is not valid.

        Returns:
            tf.Tensor: Gram matrix.
        """
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or \
                len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        gram = tf.linalg.einsum('bijc,bijd->bcd', input_layer, input_layer)
        nb_locations = tf.cast(
            tf.shape(input_layer)[1] * tf.shape(input_layer)[2], tf.float32
        )
        return gram / nb_locations

    def generate_features(self):
        """
        Extract features for style and content images.
        """
        preprocessed_style = tf.keras.applications.vgg19.preprocess_input(
            self.style_image * 255
        )
        preprocessed_content = tf.keras.applications.vgg19.preprocess_input(
            self.content_image * 255
        )

        style_outputs = self.model(preprocessed_style)[:-1]
        self.content_feature = self.model(preprocessed_content)[-1]
        self.gram_style_features = [
            self.gram_matrix(output) for output in style_outputs
        ]

    def layer_style_cost(self, style_output, gram_target):
        """
        Calculate style cost for a layer.

        Args:
            style_output (tf.Tensor): Generated image layer output.
            gram_target (tf.Tensor): Target gram matrix.

        Raises:
            TypeError: If inputs are not valid.

        Returns:
            tf.Tensor: Style cost.
        """
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or \
                len(style_output.shape) != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        c = style_output.shape[-1]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or \
                gram_target.shape != (1, c, c):
            raise TypeError(
                f"gram_target must be a tensor of shape [1, {c}, {c}]"
            )

        gram_style_output = self.gram_matrix(style_output)
        return tf.reduce_mean(tf.square(gram_style_output - gram_target))

    def style_cost(self, style_outputs):
        """
        Calculate style cost.

        Args:
            style_outputs (list): List of style layer outputs.

        Raises:
            TypeError: If style_outputs is not valid.

        Returns:
            tf.Tensor: Style cost.
        """
        if not isinstance(style_outputs, list) or len(style_outputs) != \
                len(self.style_layers):
            raise TypeError(
                f"style_outputs must be a list with a length of "
                f"{len(self.style_layers)}"
            )

        total_cost = 0
        weight = 1.0 / len(self.style_layers)
        for output, gram_target in zip(style_outputs,
                                       self.gram_style_features):
            total_cost += weight * self.layer_style_cost(output, gram_target)
        return total_cost

    def content_cost(self, content_output):
        """
        Calculate content cost.

        Args:
            content_output (tf.Tensor): Content layer output.

        Raises:
            TypeError: If content_output is not valid.

        Returns:
            tf.Tensor: Content cost.
        """
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or \
                content_output.shape != self.content_feature.shape:
            raise TypeError(
                f"content_output must be a tensor of shape "
                f"{self.content_feature.shape}"
            )

        return tf.reduce_mean(tf.square(content_output - self.content_feature))

    def total_cost(self, generated_image):
        """
        Calculate total cost for generated image.

        Args:
            generated_image (tf.Tensor): Generated image tensor.

        Raises:
            TypeError: If generated_image is not valid.

        Returns:
            tuple: Total cost, content cost, style cost.
        """
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or \
                generated_image.shape != self.content_image.shape:
            raise TypeError(
                f"generated_image must be a tensor of shape "
                f"{self.content_image.shape}"
            )

        preprocessed_generated = tf.keras.applications.vgg19.preprocess_input(
            generated_image * 255
        )
        outputs = self.model(preprocessed_generated)
        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style
