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
