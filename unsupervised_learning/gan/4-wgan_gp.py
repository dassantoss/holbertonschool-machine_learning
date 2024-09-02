#!/usr/bin/env python3
"""
Wasserstein GAN with gradient penalty and weight replacement.
"""
import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """
    This class represents a Wasserstein GAN (WGAN) with Gradient Penalty.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=.005, lambda_gp=10):
        """
        Initializes the WGAN-GP model with a generator, discriminator,
        latent generator, real examples, and other parameters.

        Parameters:
        - generator (keras.Model): The generator model.
        - discriminator (keras.Model): The discriminator model.
        - latent_generator (function): Function that generates latent vectors.
        - real_examples (tf.Tensor): Tensor containing real examples for
        training.
        - batch_size (int): The size of the batches to be used during training.
        Default is 200.
        - disc_iter (int): Number of iterations for the discriminator per
        training step. Default is 2.
        - learning_rate (float): Learning rate for the optimizers.
        Default is 0.005.
        - lambda_gp (float): The coefficient for the gradient penalty term.
        Default is 10.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = .3
        self.beta_2 = .9
        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        self.generator.loss = lambda x: -tf.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer,
                               loss=self.generator.loss)

        self.discriminator.loss = \
            lambda x, y: tf.reduce_mean(y) - tf.reduce_mean(x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer,
                                   loss=self.discriminator.loss)

    def replace_weights(self, gen_h5, disc_h5):
        """
        Replaces the weights of the generator and discriminator with those
        stored in the provided .h5 files.

        Parameters:
        - gen_h5 (str): Path to the .h5 file containing the generator's weights
        - disc_h5 (str): Path to the .h5 file containing the discriminator's
        weights.
        """
        self.generator.load_weights(gen_h5)
        self.discriminator.load_weights(disc_h5)
        print(f"Replaced generator weights from {gen_h5}")
        print(f"Replaced discriminator weights from {disc_h5}")

    def get_fake_sample(self, size=None, training=False):
        """
        Generates a batch of fake samples using the generator.

        Parameters:
        - size (int, optional): Number of fake samples to generate. If not
                                specified, the batch size specified during
                                model initialization will be used. Default is
                                None.
        - training (bool, optional): Boolean indicating whether the generator
                                     should produce samples in training mode
                                     (with dropout, etc.) or inference mode.
                                     Default is False.

        Returns:
        - tf.Tensor: A tensor containing the generated fake samples.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Retrieves a batch of real samples from the dataset.

        Parameters:
        - size (int, optional): Number of real samples to retrieve. If not
                                specified, the batch size specified during
                                model initialization will be used. Default is
                                None.

        Returns:
        - tf.Tensor: A tensor containing the real samples.
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """
        Generates interpolated samples between real and fake samples.

        Parameters:
        - real_sample (tf.Tensor): A tensor containing real samples.
        - fake_sample (tf.Tensor): A tensor containing fake samples.

        Returns:
        - tf.Tensor: A tensor containing the interpolated samples.
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """
        Computes the gradient penalty for the interpolated samples.

        Parameters:
        - interpolated_sample (tf.Tensor): A tensor containing interpolated
        samples.

        Returns:
        - tf.Tensor: The gradient penalty.
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, _):
        """
        Executes one training step for the WGAN-GP model.

        Returns:
        - dict: A dictionary containing the losses for the discriminator,
                generator, and gradient penalty.
        """
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                interpolated_sample = \
                    self.get_interpolated_sample(real_samples, fake_samples)

                real_preds = self.discriminator(real_samples, training=True)
                fake_preds = self.discriminator(fake_samples, training=True)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)

                gp = self.gradient_penalty(interpolated_sample)
                new_discr_loss = discr_loss + self.lambda_gp * gp

            discr_gradients = \
                disc_tape.gradient(new_discr_loss,
                                   self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients,
                    self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_preds = self.discriminator(fake_samples, training=False)
            gen_loss = self.generator.loss(fake_preds)

        gen_gradients = \
            gen_tape.gradient(gen_loss,
                              self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}
