#!/usr/bin/env python3
"""
Simple GAN implementation
"""
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """
    Simple GAN model with a generator and discriminator.
    """
    def __init__(self, generator, discriminator, latent_generator,
                 real_examples, batch_size=200, disc_iter=2,
                 learning_rate=0.005):
        """
        Initialize the Simple_GAN model.

        Parameters:
        - generator: Keras model for generating fake samples.
        - discriminator: Keras model for discriminating real vs fake.
        - latent_generator: Function to generate latent vectors.
        - real_examples: Tensor of real samples for training.
        - batch_size: Size of each training batch.
        - disc_iter: Discriminator updates per generator update.
        - learning_rate: Learning rate for Adam optimizer.
        """
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # Setup generator
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape))
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.generator.compile(
            optimizer=self.generator.optimizer, loss=self.generator.loss)

        # Setup discriminator
        self.discriminator.loss = lambda x, y: (
            tf.keras.losses.MeanSquaredError()(x, tf.ones(x.shape)) +
            tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape)))
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2)
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss)

    def get_fake_sample(self, size=None, training=False):
        """
        Generate fake samples using the generator.
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """
        Retrieve real samples from the dataset.
        """
        if not size:
            size = self.batch_size
        random_indices = tf.random.shuffle(
            tf.range(tf.shape(self.real_examples)[0])
        )[:size]
        return tf.gather(self.real_examples, random_indices)

    def train_step(self, _):
        """
        Perform one training step, updating both discriminator and generator.

        Returns:
        - A dictionary with discriminator and generator losses.
        """
        # Train the discriminator multiple times
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                real_samples = self.get_real_sample()
                fake_samples = self.get_fake_sample(training=True)
                real_preds = self.discriminator(real_samples)
                fake_preds = self.discriminator(fake_samples)
                discr_loss = self.discriminator.loss(real_preds, fake_preds)
            discr_gradients = disc_tape.gradient(
                discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradients, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as gen_tape:
            fake_samples = self.get_fake_sample(training=True)
            fake_preds = self.discriminator(fake_samples)
            gen_loss = self.generator.loss(fake_preds)
        gen_gradients = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables))

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
