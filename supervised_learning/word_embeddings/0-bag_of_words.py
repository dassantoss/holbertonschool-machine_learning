#!/usr/bin/env python3
"""
Bag of Words embedding matrix
"""

import string
import re
import numpy as np


def preprocess_sentences(sentences):
    """
    Preprocesses the given list of sentences by normalizing characters to
    lowercase, removing possessive "'s", and stripping punctuation.

    Args:
        sentences (list of str): List of sentences to preprocess.

    Returns:
        list of list of str: Preprocessed and tokenized sentences as a list
        of lists of words.
    """
    preprocessed_sentences = []

    for sentence in sentences:
        # Normalize the sentence to lowercase
        processed_sentence = sentence.lower()

        # Remove possessive "'s" (for example: "children's" -> "children")
        processed_sentence = re.sub(r"\'s\b", "", processed_sentence)

        # Remove remaining apostrophes and other punctuation
        processed_sentence = re.sub(
            f"[{re.escape(string.punctuation)}]", "", processed_sentence)

        # Split the sentence into words and append to the list
        preprocessed_sentences.append(processed_sentence.split())

    return preprocessed_sentences


def build_vocab(processed_sentences):
    """
    Builds a sorted vocabulary from the given preprocessed sentences.

    Args:
        processed_sentences (list of list of str): Tokenized sentences as a
        list of lists of words.

    Returns:
        list of str: A sorted list of unique words (vocabulary) from the
        preprocessed sentences.
    """
    return sorted(set(
        word for sentence in processed_sentences for word in sentence))


def bag_of_words(sentences, vocab=None):
    """
    Creates a Bag of Words embedding matrix from a list of sentences.

    Args:
        sentences (list of str): List of sentences to analyze.
        vocab (list of str, optional): A predefined list of vocabulary words
        to use for the analysis. If None, all unique words from sentences will
        be used to build the vocabulary. Defaults to None.

    Returns:
        tuple:
            embeddings (numpy.ndarray): A 2D array of shape (s, f) where:
                - s is the number of sentences.
                - f is the number of unique words (features).
                The array contains the frequency of each word
                (from the vocabulary) in each sentence.
            features (list of str): The list of features (words)
            corresponding to the columns of the embeddings matrix, sorted
            alphabetically.
    """
    # Preprocess each sentence
    processed_sentences = preprocess_sentences(sentences)

    # If vocab is not provided, build the vocabulary from the sentences
    if vocab is None:
        vocab = build_vocab(processed_sentences)

    # Mapping from each word to its index in the vocab
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Initialize an embedding matrix of zeros with shape (s, f)
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    features = np.array(vocab)

    # Fill the embedding matrix
    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, features
