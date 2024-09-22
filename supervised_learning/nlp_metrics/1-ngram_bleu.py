#!/usr/bin/env python3
"""
N-gram BLEU score calculation
"""
import numpy as np
from collections import Counter


def get_ngrams(sentence, n):
    """
    Extracts n-grams from a sentence.

    :param sentence: List of words in the sentence
    :param n: The size of the n-gram
    :return: List of n-grams as tuples
    """
    return [tuple(sentence[i:i+n]) for i in range(len(sentence) - n + 1)]


def count_matches(references, sentence_ngrams, n):
    """
    Counts the number of matching n-grams between the candidate sentence and
    the reference translations.

    :param references: List of reference translations (each translation is a
    list of words)
    :param sentence_ngrams: N-grams of the candidate sentence
    :param n: Size of the n-gram
    :return: Number of matching n-grams
    """
    sentence_ngrams_count = Counter(sentence_ngrams)

    max_ref_ngrams = Counter()
    for reference in references:
        ref_ngrams = get_ngrams(reference, n)
        ref_ngrams_count = Counter(ref_ngrams)
        for ngram in ref_ngrams_count:
            max_ref_ngrams[ngram] = max(max_ref_ngrams[ngram],
                                        ref_ngrams_count[ngram])

    matches = 0
    for ngram in sentence_ngrams_count:
        matches += min(sentence_ngrams_count[ngram],
                       max_ref_ngrams.get(ngram, 0))

    return matches


def calculate_brevity_penalty(references, sentence_len):
    """
    Calculates the brevity penalty based on sentence length and closest
    reference length.

    :param references: List of reference translations
    :param sentence_len: Length of the candidate sentence
    :return: Brevity penalty (float)
    """
    closest_ref_len = min((abs(len(ref) - sentence_len), len(ref))
                          for ref in references)[1]

    if sentence_len > closest_ref_len:
        return 1
    else:
        return np.exp(1 - closest_ref_len / sentence_len)


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a candidate sentence.

    :param references: List of reference translations (each translation is
    a list of words)
    :param sentence: List containing the model's proposed sentence
    :param n: Size of the n-gram to use for evaluation
    :return: The n-gram BLEU score
    """
    sentence_ngrams = get_ngrams(sentence, n)
    matches = count_matches(references, sentence_ngrams, n)

    total_ngrams = len(sentence_ngrams)
    precision = matches / total_ngrams if total_ngrams > 0 else 0

    brevity_penalty = calculate_brevity_penalty(references, len(sentence))

    return brevity_penalty * precision
