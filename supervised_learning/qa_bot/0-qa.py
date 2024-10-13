#!/usr/bin/env python3
"""
This module contains the question_answer function that performs
Question Answering (QA) using a pre-trained BERT model from
tensorflow-hub and transformers library.
"""
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer a question.

    Args:
        question (str): string containing the question to answer.
        reference (str): string containing the reference document from
        which to find the answer.

    Returns:
        str or None: a string containing the answer, or None if no answer
        is found.
    """
    # Load the pre-trained BERT model from TensorFlow Hub
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # Load the pre-trained tokenizer from transformers
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad'
    )

    # Tokenize the question and the reference document
    inputs = tokenizer(question, reference, return_tensors="tf")

    # Prepare the input for the TensorFlow Hub model
    input_tensors = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]

    # Pass the input tensors to the model
    output = model(input_tensors)

    # Access the start and end logits
    start_logits = output[0]
    end_logits = output[1]

    # Get the input sequence length
    sequence_length = inputs["input_ids"].shape[1]

    # Find the best start and end indices within the input sequence
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

    # Get the answer tokens using the best indices
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    # Decode the answer tokens
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # Return None if no valid answer is found
    if not answer.strip():
        return None

    return answer
