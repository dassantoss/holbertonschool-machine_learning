#!/usr/bin/env python3
"""
Wasserstein GAN with Gradient Penalty (WGAN-GP)
"""

import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    This class represents a Wasserstein GAN (WGAN) with gradient penalty.
    """

    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initializes the WGAN-GP model with a generator, discriminator,
        latent generator, real examples, and other configurations.
        """
        # Run the __init__ of keras.Model first.
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.lambda_gp = lambda_gp  # Gradient penalty coefficient

        # Dimensions and shape calculations
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # Define the generator loss and optimizer
        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=.3,
            beta_2=.9
        )
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        # Define the discriminator loss and optimizer
        self.discriminator.loss = \
            lambda x, y: tf.reduce_mean(y) - tf.reduce_mean(x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=.3,
            beta_2=.9
        )
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generates a batch of fake samples using the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size),
                              training=training)

    def get_real_sample(self, size=None):
        """
        Retrieves a batch of real samples from the dataset.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generates interpolated samples between real and fake samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Computes the gradient penalty for the interpolated samples.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, _):
        """
        Trains the WGAN-GP's discriminator and generator and returns
        their respective losses and gradient penalty. This method
        overloads the original keras.Model train_step method.

        :Returns:
        - A dictionary containing the discriminator loss, generator loss,
          and gradient penalty:
            `{"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}`
        """

        # Step 1: Train the discriminator multiple times (disc_iter)
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                interpolated_samples = \
                    self.get_interpolated_sample(real_samples,
                                                 fake_samples)

                real_preds = self.discriminator(real_samples,
                                                training=True)
                fake_preds = self.discriminator(fake_samples,
                                                training=True)

                discr_loss = self.discriminator.loss(real_preds,
                                                     fake_preds)
                gp = self.gradient_penalty(interpolated_samples)

                # Total discriminator loss including gradient penalty
                discr_loss += self.lambda_gp * gp

            discr_gradients = \
                disc_tape.gradient(discr_loss,
                                   self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables)
            )

        # Step 2: Train the generator
        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_preds = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_preds)

        gen_gradients = \
            gen_tape.gradient(gen_loss,
                              self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
