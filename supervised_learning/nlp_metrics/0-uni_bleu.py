#!/usr/bin/env python3
"""
Unigram BLEU score calculation
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a proposed sentence.

    Args:
        references (list of lists): A list of reference translations, where
            each reference is a list of words.
        sentence (list): A list of words representing the model's proposed
            sentence.

    Returns:
        float: The unigram BLEU score for the proposed sentence.
    """
    sentence_len = len(sentence)

    # Calculate maximum matches for each word in the sentence
    matches = 0
    for word in set(sentence):
        max_count = max(ref.count(word) for ref in references)
        matches += min(sentence.count(word), max_count)

    # Calculate precision (correct matches / total words in sentence)
    precision = matches / sentence_len

    # Find the reference length closest to the sentence length
    closest_ref_len = min(
        (abs(len(ref) - sentence_len), len(ref)) for ref in references)[1]

    # Calculate brevity penalty
    if sentence_len > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / sentence_len)

    # Return the BLEU score (brevity penalty * precision)
    return brevity_penalty * precision
