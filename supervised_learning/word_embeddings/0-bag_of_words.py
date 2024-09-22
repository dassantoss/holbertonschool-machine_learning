#!/usr/bin/env python3
"""
Bag of Words embedding matrix
"""
import string
import re
import numpy as np


def preprocess_sentences(sentences):
    """
    Preprocesses sentences by converting to lowercase, removing possessives,
    and stripping punctuation.

    Args:
        sentences (list of str): Sentences to process.

    Returns:
        list of list of str: Tokenized and cleaned sentences.
    """
    preprocessed_sentences = []

    for sentence in sentences:
        # Normalize the sentence to lowercase
        processed_sentence = sentence.lower()

        # Remove possessive "'s"
        processed_sentence = re.sub(r"\'s\b", "", processed_sentence)

        # Remove punctuation
        processed_sentence = re.sub(
            f"[{re.escape(string.punctuation)}]", "", processed_sentence)

        # Split into words
        preprocessed_sentences.append(processed_sentence.split())

    return preprocessed_sentences


def build_vocab(processed_sentences):
    """
    Builds a sorted vocabulary from processed sentences.

    Args:
        processed_sentences (list of list of str): Tokenized sentences.

    Returns:
        list of str: Sorted list of unique words (vocabulary).
    """
    return sorted(
        set(word for sentence in processed_sentences for word in sentence))


def bag_of_words(sentences, vocab=None):
    """
    Creates a Bag of Words embedding matrix.

    Args:
        sentences (list of str): List of sentences to analyze.
        vocab (list of str, optional): Vocabulary to use. If None,
                                       it's generated.

    Returns:
        tuple:
            - embeddings (numpy.ndarray): Embedding matrix of shape (s, f),
              where `s` is the number of sentences and `f` is the number of
              features.
            - features (list of str): The vocabulary (features) used for
                                      embeddings.
    """
    # Preprocess sentences
    processed_sentences = preprocess_sentences(sentences)

    # Build vocab if not provided
    if vocab is None:
        vocab = build_vocab(processed_sentences)

    # Create word-to-index mapping
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # Initialize embeddings matrix
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Fill the embeddings matrix
    for i, sentence in enumerate(processed_sentences):
        for word in sentence:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, vocab
