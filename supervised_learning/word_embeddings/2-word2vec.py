#!/usr/bin/env python3
"""
Word2Vec model training
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model using the Gensim library.

    Args:
        sentences (list of list of str): List of tokenized sentences to be
            used for training.
        vector_size (int, optional): Dimensionality of the embedding layer.
            Defaults to 100.
        min_count (int, optional): Minimum number of occurrences of a word
            to be considered for training. Words with fewer occurrences are
            ignored. Defaults to 5.
        window (int, optional): Maximum distance between the current and
            predicted word within a sentence. Defaults to 5.
        negative (int, optional): Size of negative sampling. Defaults to 5.
        cbow (bool, optional): If True, Continuous Bag of Words (CBOW) is
            used for training. If False, Skip-gram is used. Defaults to True.
        epochs (int, optional): Number of iterations (epochs) to train the
            model. Defaults to 5.
        seed (int, optional): Seed for the random number generator to ensure
            reproducibility. Defaults to 0.
        workers (int, optional): Number of worker threads used during
            training. Defaults to 1.

    Returns:
        gensim.models.Word2Vec: The trained Word2Vec model.
    """
    # Choose training algorithm: CBOW (sg=0) or Skip-gram (sg=1)
    sg = 0 if cbow else 1

    # Initialize the Word2Vec model with the specified parameters
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    # Build vocabulary from the sentences and train the model
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
