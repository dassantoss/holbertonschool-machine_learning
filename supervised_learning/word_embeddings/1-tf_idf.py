#!/usr/bin/env python3
"""
TF-IDF embedding matrix creation
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding matrix from a list of sentences.

    Args:
        sentences (list of str): List of sentences to analyze.
        vocab (list of str, optional): List of vocabulary words to use for
        the analysis. If None, all words within the sentences will be used.

    Returns:
        tuple:
            embeddings (numpy.ndarray): A 2D array of shape (s, f) where:
                - s is the number of sentences.
                - f is the number of features (unique words in the vocab).
              Each element in the array contains the TF-IDF score of a word
              in the corresponding sentence.
            features (list of str): The list of features (words)
              corresponding to the columns of the embeddings matrix.
    """
    # Initialize the TF-IDF vectorizer with the given vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform the sentences to get the TF-IDF embeddings
    embeddings = vectorizer.fit_transform(sentences)

    # Extract the features (words) used by the vectorizer
    features = vectorizer.get_feature_names_out()

    return embeddings.toarray(), features
