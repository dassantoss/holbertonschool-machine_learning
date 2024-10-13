#!/usr/bin/env python3
"""
This module contains the answer_loop function that continuously interacts
with the user, answering questions based on a reference text.
If the answer cannot be found, it responds with a default message.
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """
    Continuosly answers questions from a reference text.
    Exits the loop if the user types 'exit', 'quit', 'goodbye', or 'bye'.

    Args:
        reference (str): The reference text from which to find answers.
    """
    while True:
        # Prompt user for input with 'Q:'
        question = input("Q: ").strip()

        # Convert the input to lowercase for comparison
        question_lower = question.lower()

        # Check if the user wants to exit
        if question_lower in ['exit', 'quit', 'goodbye', 'bye']:
            print("A: Goodbye")
            break

        # Get the answer from the question_answer function
        answer = question_answer(question, reference)

        # If no answer is found, return a default message
        if answer is None:
            print("A: Sorry, I do not understand your question.")
            print("Hint: Maybe the question isn't covered in the reference.")
        else:
            print(f"A: {answer}")
