# Natural Language Processing - Word Embeddings

## Project Overview

This project is part of Holberton School's curriculum on **Machine Learning** and focuses on **Natural Language Processing (NLP)**, specifically **word embeddings**. Word embeddings are a way to represent words in vector space, allowing machine learning models to process and understand text data. Throughout this project, we implement and work with several word embedding techniques including **Bag of Words**, **TF-IDF**, **Word2Vec**, **FastText**, and **ELMo**.

## Technologies Used

- **Python** (v3.9)
- **Gensim** (v4.3.3) for Word2Vec and FastText models
- **TensorFlow/Keras** (v2.15.0) for embedding layers and neural networks
- **NumPy** (v1.25.2) for numerical operations
- **scikit-learn** for TF-IDF vectorizer and other utilities
- **Ubuntu 20.04 LTS**

## Setup and Installation

Follow these steps to set up the project environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/holbertonschool-machine_learning
    cd supervised_learning/word_embeddings
    ```

2. Install the necessary Python packages:
    ```bash
    pip install gensim==4.3.3 tensorflow==2.15.0 numpy==1.25.2 scikit-learn
    ```

3. Verify that **Keras** is version 2.15.0:
    ```bash
    python3 -c 'import keras; print(keras.__version__)'
    ```

## Tasks Implemented

### Task 0: Bag of Words

- **File**: `0-bag_of_words.py`
- **Description**: Implements the **Bag of Words** model. This model creates an embedding matrix representing the frequency of each word in the given sentences.
- **Key Concepts**: Word occurrence counts.

    ```bash
    ./0-main.py
    [[0 1 0 0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0]
    ...
    ['are', 'awesome', 'beautiful', 'cake', 'children', 'future', 'good', 'grandchildren', 'holberton', 'is', 'learning', 'life', 'machine', 'nlp', 'no', 'not', 'one', 'our', 'said', 'school', 'that', 'the', 'very', 'was']
    ```

### Task 1: TF-IDF

- **File**: `1-tf_idf.py`
- **Description**: Implements **TF-IDF (Term Frequency-Inverse Document Frequency)** embedding. This method measures the importance of each word relative to all the sentences in the corpus.
- **Key Concepts**: Word importance based on frequency across documents.

    ```bash
    ./1-main.py
    [[1.         0.         0.         0.         0.         0.        ]
    ...
    ['awesome', 'learning', 'children', 'cake', 'good', 'none', 'machine']
    ```

### Task 2: Train Word2Vec Model

- **File**: `2-word2vec.py`
- **Description**: Creates and trains a **Word2Vec** model using **CBOW** (Continuous Bag of Words) or **Skip-gram** method, depending on the user’s choice.
- **Key Concepts**: Contextual word prediction using word vectors.

    ```bash
    ./2-main.py
    [['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
    [-5.4084123e-03 -4.0024161e-04 -3.4630739e-03 ...]
    ```

### Task 3: Convert Word2Vec to Keras Embedding Layer

- **File**: `3-gensim_to_keras.py`
- **Description**: Converts a trained Word2Vec model into a Keras **Embedding** layer that can be used in neural network models.
- **Key Concepts**: Utilizing pre-trained embeddings in deep learning models.

    ```bash
    ./3-main.py
    <keras.src.layers.core.embedding.Embedding object at 0x7f08126b8910>
    ```bash

### Task 4: FastText Model

- **File**: `4-fasttext.py`
- **Description**: Builds and trains a **FastText** model, which extends the Word2Vec model to work with subwords, making it more robust for handling rare or misspelled words.
- **Key Concepts**: Subword information in embeddings.

    ```bash
    ./4-main.py
    [['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time']]
    [-4.4518875e-04  1.9057443e-04  7.1344204e-04 ...]
    ```

### Task 5: ELMo Quiz

- **File**: `5-elmo`
- **Description**: A multiple-choice quiz to test understanding of the components trained in an **ELMo (Embeddings from Language Models)** model.
- **Answer**: A (Training involves the BiLSTM weights, character embeddings, and weights applied to hidden states).

## Running the Project

To run the project, execute the main Python file corresponding to each task:

For **Task 0** (Bag of Words):

    ```bash
    ./0-main.py
    ```

For **Task 1** (TF-IDF):

    ```bash
    ./1-main.py
    ```

For **Task 2** (Word2Vec):

    ```bash
    ./2-main.py
    ```

For **Task 3** (Convert Word2Vec to Keras Embedding Layer):

    ```bash
    ./3-main.py
    ```

For **Task 4** (FastText):

    ```bash
    ./4-main.py
    ```

## References

- **Efficient Estimation of Word Representations in Vector Space (Skip-gram, 2013)**
- **Distributed Representations of Words and Phrases and their Compositionality (Word2Vec, 2013)**
- **GloVe: Global Vectors for Word Representation (2014)**
- **fastText (2016)**, Bag of Tricks for Efficient Text Classification
- **ELMo (2018)**, Deep contextualized word representations
