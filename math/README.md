## Tasks

0. **Slice Me Up** - Slicing arrays in various ways.
Complete the following source code (found below):
- `arr1` should be the first two numbers of `arr`
- `arr2` should be the last five numbers of `arr`
- `arr3` should be the 2nd through 6th numbers of `arr`
- You are not allowed to use any loops or conditional statements
- Your program should be exactly 8 lines

The program should consist of exactly 8 lines.

```python
alexa@ubuntu-xenial:linear_algebra$ cat 0-slice_me_up.py 
#!/usr/bin/env python3
arr = [9, 8, 2, 3, 9, 4, 1, 0, 3]
arr1 =  # your code here
arr2 =  # your code here
arr3 =  # your code here
print("The first two numbers of the array are: {}".format(arr1))
print("The last five numbers of the array are: {}".format(arr2))
print("The 2nd through 6th numbers of the array are: {}".format(arr3))
alexa@ubuntu-xenial:linear_algebra$ ./0-slice_me_up.py 
The first two numbers of the array are: [9, 8]
The last five numbers of the array are: [9, 4, 1, 0, 3]
The 2nd through 6th numbers of the array are: [8, 2, 3, 9, 4]
alexa@ubuntu-xenial:linear_algebra$ wc -l 0-slice_me_up.py 
8 0-slice_me_up.py
alexa@ubuntu-xenial:linear_algebra$ 
```

## Repo:
- GitHub repository: holbertonschool-machine_learning
- Directory: math/linear_algebra
- File: 0-slice_me_up.py

1. **Trim Me Down** - Extracting specific columns from a matrix.
Complete the following source code (found below):

- `the_middle` should be a 2D matrix containing the 3rd and 4th columns of `matrix`
- You are not allowed to use any conditional statements
- You are only allowed to use one `for` loop
- Your program should be exactly 6 lines

```python
alexa@ubuntu-xenial:linear_algebra$ cat 1-trim_me_down.py 
#!/usr/bin/env python3
matrix = [[1, 3, 9, 4, 5, 8], [2, 4, 7, 3, 4, 0], [0, 3, 4, 6, 1, 5]]
the_middle = []
# your code here
print("The middle columns of the matrix are: {}".format(the_middle))
alexa@ubuntu-xenial:linear_algebra$ ./1-trim_me_down.py 
The middle columns of the matrix are: [[9, 4], [7, 3], [4, 6]]
alexa@ubuntu-xenial:linear_algebra$ wc -l 1-trim_me_down.py 
6 1-trim_me_down.py
alexa@ubuntu-xenial:linear_algebra$ 
```

## Repo:
- GitHub repository: holbertonschool-machine_learning
- Directory: math/linear_algebra
- File: 1-trim_me_down.py


2. **Size Me Please** - Calculating the shape of a matrix.
Write a function `def matrix_shape(matrix):` that calculates the shape of a matrix:

- You can assume all elements in the same dimension are of the same type/shape
- The shape should be returned as a list of integers

```python
alexa@ubuntu-xenial:linear_algebra$ cat 2-main.py 
#!/usr/bin/env python3

matrix_shape = __import__('2-size_me_please').matrix_shape

mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))
mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_shape(mat2))
alexa@ubuntu-xenial:linear_algebra$ ./2-main.py 
[2, 2]
[2, 3, 5]
alexa@ubuntu-xenial:linear_algebra$ 
```

## Repo:
- GitHub repository: holbertonschool-machine_learning
- Directory: math/linear_algebra
- File: 2-size_me_please.py

3. **Flip Me Over** - Transposing a matrix.
Write a function `def matrix_transpose(matrix):` that returns the transpose of a 2D matrix, `matrix`:

- You must return a new matrix
- You can assume that `matrix` is never empty
- You can assume all elements in the same dimension are of the same type/shape

```python
alexa@ubuntu-xenial:linear_algebra$ cat 3-main.py 
#!/usr/bin/env python3

matrix_transpose = __import__('3-flip_me_over').matrix_transpose

mat1 = [[1, 2], [3, 4]]
print(mat1)
print(matrix_transpose(mat1))
mat2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
        [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
print(mat2)
print(matrix_transpose(mat2))
alexa@ubuntu-xenial:linear_algebra$ ./3-main.py 
[[1, 2], [3, 4]]
[[1, 3], [2, 4]]
[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
[[1, 6, 11, 16, 21, 26], [2, 7, 12, 17, 22, 27], [3, 8, 13, 18, 23, 28], [4, 9, 14, 19, 24, 29], [5, 10, 15, 20, 25, 30]]
alexa@ubuntu-xenial:linear_algebra$ 
```

## Repo:
- GitHub repository: holbertonschool-machine_learning
- Directory: math/linear_algebra
- File: 3-flip_me_over.py

4. **Line Up** - Adding two arrays element-wise.
Write a function `def add_arrays(arr1, arr2):` that adds two arrays element-wise:

- You can assume that `arr1` and `arr2` are lists of ints/floats
- You must return a new list
- If `arr1` and `arr2` are not the same shape, return `None`

```python
alexa@ubuntu-xenial:linear_algebra$ cat 4-main.py 
#!/usr/bin/env python3

add_arrays = __import__('4-line_up').add_arrays

arr1 = [1, 2, 3, 4]
arr2 = [5, 6, 7, 8]
print(add_arrays(arr1, arr2))
print(arr1)
print(arr2)
print(add_arrays(arr1, [1, 2, 3]))
alexa@ubuntu-xenial:linear_algebra$ ./4-main.py 
[6, 8, 10, 12]
[1, 2, 3, 4]
[5, 6, 7, 8]
None
alexa@ubuntu-xenial:linear_algebra$ 
```

## Repo:
- GitHub repository: holbertonschool-machine_learning
- Directory: math/linear_algebra
- File: 4-line_up.py

5. **Across The Planes** - Adding two matrices element-wise.
Write a function `def add_matrices2D(mat1, mat2):` that adds two matrices element-wise:

- You can assume that `mat1` and `mat2` are 2D matrices containing ints/floats
- You can assume all elements in the same dimension are of the same type/shape
- You must return a new matrix
- If `mat1` and `mat2` are not the same shape, return `None`

6. **Howdy Partner** - Concatenating two arrays.
7. **Gettin’ Cozy** - Concatenating two matrices along a specific axis.
8. **Ridin’ Bareback** - Performing matrix multiplication.
9. **Let The Butcher Slice It** - Complex slicing on matrices.
10. **I’ll Use My Scale** - Determining the shape of a NumPy array.
11. **The Western Exchange** - Transposing a NumPy array.
12. **Bracing The Elements** - Element-wise operations on NumPy arrays.
13. **Cat's Got Your Tongue** - Concatenating NumPy arrays along a specific axis.
14. **Saddle Up** - Matrix multiplication using NumPy.
15. **Slice Like A Ninja** (Advanced) - Advanced slicing techniques on NumPy arrays.
16. **The Whole Barn** (Advanced) - Adding matrices of varying dimensions.
17. **Squashed Like Sardines** (Advanced) - Advanced matrix concatenation.

# This README is under construction.
