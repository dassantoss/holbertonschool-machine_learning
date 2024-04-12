#!/usr/bin/env python3
"""
Determine the shape of a matrix as a list of its dimensions.
"""
def matrix_shape(matrix):
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
