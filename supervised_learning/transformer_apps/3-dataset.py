#!/usr/bin/env python3
"""
Pipeline for machine translation dataset preparation
"""

import tensorflow_datasets as tfds
import transformers
import tensorflow as tf


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset
    for machine translation from Portuguese to English.
    """

    def __init__(self, batch_size, max_len):
        """
        Initializes the Dataset object and sets up the data pipeline.

        Args:
            batch_size: the batch size for training/validation.
            max_len: the maximum number of tokens allowed per example sentence.
        """
        # Load the Portuguese to English translation dataset
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Initialize tokenizers using the training data
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Tokenize the dataset using eager execution
        self.data_train = \
            self.data_train.map(self.tf_encode,
                                num_parallel_calls=tf.data.AUTOTUNE)
        self.data_valid = \
            self.data_valid.map(self.tf_encode,
                                num_parallel_calls=tf.data.AUTOTUNE)

        # Store max_len and batch_size
        self.max_len = max_len
        self.batch_size = batch_size

        # Prepare the data pipeline for training and validation datasets
        self.data_train = self.prepare_pipeline(self.data_train,
                                                is_training=True)
        self.data_valid = self.prepare_pipeline(self.data_valid,
                                                is_training=False)

    def filter_max_len(self, sentence_1, sentence_2):
        """
        Filters out examples where either sentence exceeds max_len.

        Args:
            sentence_1: Portuguese sentence.
            sentence_2: English sentence.

        Returns:
            A boolean tensor indicating if both sentences are within max_len.
        """
        return tf.logical_and(tf.size(sentence_1) <= self.max_len,
                              tf.size(sentence_2) <= self.max_len)

    def prepare_pipeline(self, data, is_training):
        """
        Prepares the data pipeline by filtering, tokenizing, batching, and
        prefetching the dataset.

        Args:
            data: The dataset to process.
            is_training: Whether the dataset is for training or validation.

        Returns:
            The processed dataset.
        """
        # Filter out sentences longer than max_len
        data = data.filter(self.filter_max_len)

        if is_training:
            # Cache, shuffle, batch, and prefetch the training dataset
            data = data.cache()
            data = data.shuffle(buffer_size=20000)

        # Group into padded batches
        data = data.padded_batch(self.batch_size,
                                 padded_shapes=([None], [None]))

        # Prefetch the dataset
        data = data.prefetch(tf.data.AUTOTUNE)

        return data

    def tokenize_dataset(self, data):
        """
        Tokenizes the dataset using pre-trained tokenizers.

        Args:
            data: tf.data.Dataset containing tuples of (pt, en) sentences.

        Returns:
            tokenizer_pt, tokenizer_en: Tokenizers for Portuguese and English.
        """
        pt_sentences = []
        en_sentences = []
        for pt, en in data.as_numpy_iterator():
            pt_sentences.append(pt.decode('utf-8'))
            en_sentences.append(en.decode('utf-8'))

        # Load the pre-trained tokenizers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        # Train both tokenizers on the dataset sentences
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(pt_sentences,
                                                            vocab_size=2**13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(en_sentences,
                                                            vocab_size=2**13)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """
        Encodes a translation into tokens, including start and end of sentence
        tokens.

        Args:
            pt: Portuguese sentence.
            en: English sentence.

        Returns:
            pt_tokens, en_tokens: Encoded sentences as arrays of tokens.
        """
        # Decode tensors to strings
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        # Get the vocabulary sizes from the tokenizers
        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        # Tokenize sentences without special tokens
        pt_tokens = self.tokenizer_pt.encode(pt_sentence,
                                             add_special_tokens=False)
        en_tokens = self.tokenizer_en.encode(en_sentence,
                                             add_special_tokens=False)

        # Add start and end tokens
        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """
        TensorFlow wrapper for the encode method to ensure compatibility with
        TensorFlow's eager execution.

        Args:
            pt: Portuguese sentence.
            en: English sentence.

        Returns:
            pt_tokens, en_tokens: Encoded sentences as tensors.
        """
        pt_tokens, en_tokens = tf.py_function(func=self.encode,
                                              inp=[pt, en],
                                              Tout=[tf.int64, tf.int64])

        # Set the shapes of the output tensors
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
