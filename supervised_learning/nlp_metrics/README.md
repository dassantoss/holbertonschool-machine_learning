# Natural Language Processing - Evaluation Metrics

## Project Overview

This project is part of the **Natural Language Processing (NLP)** module within Holberton School's curriculum. The focus is on implementing and understanding various **evaluation metrics** used to evaluate the performance of NLP models, specifically **BLEU** (Bilingual Evaluation Understudy) scores. These metrics are important in assessing how well a machine translation model or other text generation models perform in relation to a set of reference translations.

## Resources

- 7 Applications of Deep Learning for Natural Language Processing
- 10 Applications of Artificial Neural Networks in Natural Language Processing
- A Gentle Introduction to Calculating the BLEU Score for Text in Python
- Bleu Score
- Evaluating Text Output in NLP: BLEU at your own risk
- ROUGE metric
- Evaluation and Perplexity
- Evaluation metrics

## Learning Objectives

At the end of this project, you should be able to:

- Explain the applications of natural language processing (NLP)
- Understand and compute the BLEU score
- Understand the ROUGE score
- Explain what perplexity is and how it’s used
- Know when to use one evaluation metric over another

## Requirements

- All files are interpreted/compiled on Ubuntu 20.04 LTS using Python 3.9
- Your files will be executed with numpy (version 1.25.2)
- Pycodestyle (version 2.11.1) is followed strictly
- No usage of the nltk module is allowed
- All files should have appropriate documentation and should end with a new line

## Tasks

### Task 0: Unigram BLEU score

- **File**: `0-uni_bleu.py`
- **Description**: This function calculates the **unigram BLEU score** for a given sentence in comparison to reference translations. A unigram BLEU score compares individual words (unigrams).
- **Usage**:

```bash
$ cat 0-main.py
#!/usr/bin/env python3

uni_bleu = __import__('0-uni_bleu').uni_bleu

references = [["the", "cat", "is", "on", "the", "mat"], 
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(uni_bleu(references, sentence))
$ ./0-main.py
0.6549846024623855
```

### Task 1: N-gram BLEU score

- **File**: `1-ngram_bleu.py`
- **Description**: This function calculates the **n-gram BLEU score** for a sentence using an arbitrary size of n-grams (pairs, triples, etc.). The BLEU score measures the precision of n-gram matches between the proposed sentence and the reference translations.
- **Usage**:

```bash
$ cat 1-main.py
#!/usr/bin/env python3

ngram_bleu = __import__('1-ngram_bleu').ngram_bleu

references = [["the", "cat", "is", "on", "the", "mat"], 
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(ngram_bleu(references, sentence, 2))
$ ./1-main.py
0.6140480648084865
```

### Task 2: Cumulative N-gram BLEU score

- **File**: `2-cumulative_bleu.py`
- **Description**: This function calculates the **cumulative n-gram BLEU score** for a sentence. It aggregates BLEU scores for all n-grams up to a given `n`, averaging them to provide a more holistic evaluation of the sentence.
- **Usage**:

```bash
$ cat 2-main.py
#!/usr/bin/env python3

cumulative_bleu = __import__('2-cumulative_bleu').cumulative_bleu

references = [["the", "cat", "is", "on", "the", "mat"], 
              ["there", "is", "a", "cat", "on", "the", "mat"]]
sentence = ["there", "is", "a", "cat", "here"]

print(cumulative_bleu(references, sentence, 4))
$ ./2-main.py
0.5475182535069453
```

```bash
Repo:

GitHub repository: holbertonschool-machine_learning
Directory: supervised_learning/nlp_metrics
```