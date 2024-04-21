#!/usr/bin/env python3

poly_derivative = __import__('10-matisse').poly_derivative

poly = [5, 3, 0, 1]
print(poly_derivative(poly))
print(poly_derivative("not a list"))  # Should return None
print(poly_derivative([5, 3, 'a', 1]))  # Should return None
print(poly_derivative([5, 3, 0, 1.5]))  # Should return None
