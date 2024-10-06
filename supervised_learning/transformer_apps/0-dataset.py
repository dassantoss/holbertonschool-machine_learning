#!/usr/bin/env python3
"""Module that creates and prepares a dataset for machine translation"""
import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    A class to load and prepare the TED HRLR translation dataset
    for machine translation from Portuguese to English.
    """

    def __init__(self):
        """
        Initializes the dataset and tokenizers.
        Loads the train and validation splits from the TED HRLR dataset
        and initializes the tokenizers for Portuguese and English.
        """
        # Load the TED HRLR dataset: Portuguese to English translation
        self.data_train = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='train', as_supervised=True)
        self.data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                                    split='validation', as_supervised=True)

        # Tokenizers initialized using pre-trained models
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset()

    def tokenize_dataset(self):
        """
        Loads pre-trained tokenizers for Portuguese and English.
        No additional training is done since the tokenizers are pre-trained.

        Returns:
        - tokenizer_pt: Pre-trained tokenizer for Portuguese.
        - tokenizer_en: Pre-trained tokenizer for English.
        """
        # Pre-trained Portuguese tokenizer
        tokenizer_pt = AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            model_max_length=2**13,
            use_fast=True,
            clean_up_tokenization_spaces=True
        )

        # Pre-trained English tokenizer
        tokenizer_en = AutoTokenizer.from_pretrained(
            'bert-base-uncased',
            model_max_length=2**13,
            use_fast=True,
            clean_up_tokenization_spaces=True
        )

        return tokenizer_pt, tokenizer_en
