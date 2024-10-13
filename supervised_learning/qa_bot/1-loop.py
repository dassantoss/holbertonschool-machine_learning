#!/usr/bin/env python3
"""
This script implements a simple interactive loop for a Question-Answer bot.
It keeps prompting the user for input with 'Q:', and prints 'A:' for the
response. The loop stops when the user enters 'exit', 'quit', 'goodbye',
or 'bye' (case insensitive).
"""


def qa_loop():
    """
    Creates an interactive question-answer loop with the user.
    Exits the loop when the user types 'exit', 'quit', 'goodbye', or 'bye'.
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

        # Simulate an empty answer for now
        print("A:")

# Entry point for the script
if __name__ == "__main__":
    qa_loop()
