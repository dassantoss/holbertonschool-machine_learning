#!/usr/bin/env python3
"""
Neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Class that performs tasks for neural style transfer
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Class constructor for neural style transfer
        """

        if (not isinstance(style_image, np.ndarray)
                or style_image.shape[-1] != 3):
            raise TypeError("style_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.style_image = self.scale_image(style_image)

        if (not isinstance(content_image, np.ndarray)
                or content_image.shape[-1] != 3):
            raise TypeError("content_image must be a numpy.ndarray"
                            " with shape (h, w, 3)")
        else:
            self.content_image = self.scale_image(content_image)

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        else:
            self.alpha = alpha

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        else:
            self.beta = beta

        self.model = None
        self.load_model()
        self.gram_style_features, self.content_feature = \
            self.generate_features()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 px
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise (TypeError
                   ("image must be a numpy.ndarray with shape (h, w, 3)"))

        h, w, _ = image.shape

        if w > h:
            w_new = 512
            h_new = int((h * 512) / w)
        else:
            h_new = 512
            w_new = int((w * 512) / h)

        resized_image = tf.image.resize(image,
                                        size=[h_new, w_new],
                                        method='bicubic')

        # Normalize
        resized_image = resized_image / 255.0

        # Limit pixel values between 0 and 1
        resized_image = tf.clip_by_value(resized_image, 0, 1)

        tf_resize_image = tf.expand_dims(resized_image, 0)

        return tf_resize_image

    def load_model(self):
        """
        Create the model used to calculate cost
        """
        # Keras API
        modelVGG19 = tf.keras.applications.VGG19(
                include_top=False,
                weights='imagenet'
                )

        modelVGG19.trainable = False

        # selected layers
        selected_layers = self.style_layers + [self.content_layer]

        outputs = [
                modelVGG19.get_layer(name).output for name in selected_layers
                ]

        # construct model
        model = tf.keras.Model([modelVGG19.input], outputs)

        # for replace MaxPooling layer by AveragePooling layer
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        tf.keras.models.save_model(model, 'vgg_base.h5')
        model_avg = tf.keras.models.load_model(
                'vgg_base.h5', custom_objects=custom_objects
                )

        self.model = model_avg

    @staticmethod
    def gram_matrix(input_layer):
        """
        Calculates the gram matrix of an input layer
        """
        if not isinstance(
                input_layer,
                (tf.Tensor, tf.Variable)
                ) or len(input_layer.shape) != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        # Get the dimensions
        _, h, w, c = input_layer.shape

        # Reshape the tensor to a 2D matrix
        input_layer_reshaped = tf.reshape(input_layer, (h * w, c))

        # Compute the gram matrix
        gram = tf.matmul(input_layer_reshaped,
                         input_layer_reshaped,
                         transpose_a=True)

        # Normalize the gram matrix
        gram_matrix = tf.expand_dims(gram / tf.cast(h * w, tf.float32), axis=0)

        return gram_matrix

    def generate_features(self):
        """
        method extract the features used to calculate neural style cost
        """
        # preprocess style and content image
        preprocess_style = (tf.keras.applications.vgg19.
                            preprocess_input(self.style_image * 255))
        preprocess_content = (
                tf.keras.applications.vgg19.
                preprocess_input(self.content_image * 255))

        # get style and content outputs from VGG19 model
        style_output = self.model(preprocess_style)
        content_output = self.model(preprocess_content)

        # compute Gram matrices for style features
        self.gram_style_features = [self.gram_matrix(style_layer) for
                                    style_layer in style_output]

        # excluding the last element considered more suitable for capturing
        # the style of image
        self.gram_style_features = self.gram_style_features[:-1]

        # select only last network layer
        self.content_feature = content_output[-1]

        return self.gram_style_features, self.content_feature

    def layer_style_cost(self, style_output, gram_target):
        """
        method to calculate the style cost for a single layer
        """

        if (not isinstance(style_output, (tf.Tensor, tf.Variable))
                or len(style_output.shape) != 4):
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape

        if (not isinstance(gram_target, (tf.Tensor, tf.Variable))
                or gram_target.shape != [1, c, c]):
            raise TypeError(
                    "gram_target must be a tensor of shape [1, {}, {}]".format(
                        c,
                        c
                        ))

        output_gram_style = self.gram_matrix(style_output)

        # difference between two gram matrix
        layer_style_cost = tf.reduce_mean(
                tf.square(output_gram_style - gram_target))

        return layer_style_cost

    def style_cost(self, style_outputs):
        """
        method to calculate style cost for generate image
        """
        len_style_layer = len(self.style_layers)
        if (not isinstance(style_outputs, list)
                or len(style_outputs) != len(self.style_layers)):
            raise TypeError(
                    "style_outputs must be a list with a length of {}"
                    .format(len_style_layer)
                    )

        # uniform initialization
        weight = 1.0 / len_style_layer

        cost_total = sum([weight * self.layer_style_cost(style, target)
                          for style, target
                          in zip(style_outputs, self.gram_style_features)])

        return cost_total

    def content_cost(self, content_output):
        """
        method calculate content cost for the generated image
        """
        content_feature_shape = self.content_feature.shape

        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != self.content_feature.shape):
            raise TypeError(
                    "content_output must be a tensor of shape {}".
                    format(content_feature_shape))

        content_cost = (
                tf.reduce_mean(tf.square(
                    content_output - self.content_feature
                    )))

        return content_cost

    def total_cost(self, generated_image):
        """
            method calculate total cost for the generated image

        :param generated_image: tf.Tensor, shape(1,nh,nw,3) generated image

        :return: (J, J_content, J_style)
                J: total cost
                J_content: content cost
                J_style: style cost
        """
        shape_content_image = self.content_image.shape

        if (not isinstance(generated_image, tf.Tensor)
                or generated_image.shape != shape_content_image):
            raise TypeError("generated_image must be a tensor of shape {}"
                            .format(shape_content_image))

        # preprocess generated img
        preprocess_generated_image = \
            (tf.keras.applications.
             vgg19.preprocess_input(generated_image * 255))

        # calculate content and style for generated image
        generated_output = self.model(preprocess_generated_image)

        # def content and style
        generated_content = generated_output[-1]
        generated_style = generated_output[:-1]

        J_content = self.content_cost(generated_content)
        J_style = self.style_cost(generated_style)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """
        method to calculate gradients for tf.tensor generated image
        shape(1,nh,nw,3)
        """
        # define shape
        shape_content_image = \
            (1, self.content_image.shape[1],
             self.content_image.shape[2], 3)

        if (not isinstance(generated_image, (tf.Tensor, tf.Variable))
                or generated_image.shape != shape_content_image):
            raise TypeError("generated_image must be a tensor of shape {}"
                            .format(shape_content_image))

        # create GradientTape context to track operations for automatic
        # differentiation
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        # calculate gradients of the total cost with respect to generated image
        # using gradient method of tape
        grad = tape.gradient(J_total, generated_image)

        return grad, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None,
                       lr=0.01, beta1=0.9, beta2=0.99):
        """
        method to generate the neural style network transferred image
        """

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and (step < 0 or step > iterations):
            raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (float, int)):
            raise TypeError("lr must be a number")
        if lr < 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("beta2 must be in the range [0, 1]")

        # initialize image
        generated_image = tf.Variable(self.content_image)

        # initialize best cost and best image
        best_cost = float('inf')
        best_image = None

        # Initialize Adam
        optimizer = tf.optimizers.Adam(lr, beta1, beta2)

        # Optimization loop
        for i in range(iterations + 1):
            # compute gradients and costs
            grads, J_total, J_content, J_style = self.compute_grads(
                    generated_image
                    )

            # use opt
            optimizer.apply_gradients([(grads, generated_image)])

            # selected best cost and best image
            if J_total < best_cost:
                best_cost = float(J_total)
                best_image = generated_image

            # Print step required
            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, style {}"
                      .format(i, J_total, J_content, J_style))

        # remove sup dim
        best_image = best_image[0]
        best_image = tf.clip_by_value(best_image, 0, 1)
        best_image = best_image.numpy()

        return best_image, best_cost
