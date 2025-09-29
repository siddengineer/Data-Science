create_2d_array.py
🔹 What is a 2D array?

A 2D array is like a table (rows × columns).

In Python, it can be represented as:

List of lists (basic way).

NumPy array (efficient for numerical tasks).

🔹 Why use a 2D array?

To store structured data (like a matrix, grid, image pixels).

Examples:

Game boards (Tic-Tac-Toe, Sudoku).

Mathematical matrices.

Data tables.

🔹 Methods to Create a 2D Array

Using list comprehension (pure Python):

rows, cols = 3, 4
arr = [[0 for _ in range(cols)] for _ in range(rows)]
print(arr)
# Output: [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]


Using NumPy (better for math/DS):

import numpy as np
arr = np.zeros((3, 4), dtype=int)
print(arr)
# Output: 3x4 array of zeros


Manual creation:

arr = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print(arr)

🔹 Accessing Elements
print(arr[0][2])  # first row, third column

🔹 Modifying Elements
arr[1][1] = 99
print(arr)

📂 Code: create_2d_array.py
# create_2d_array.py

# 1. Create a 2D array using list comprehension
rows, cols = 3, 4
array_2d = [[0 for _ in range(cols)] for _ in range(rows)]
print("2D Array using list comprehension:")
print(array_2d)

# 2. Create a 2D array manually
manual_array = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
print("\nManual 2D Array:")
print(manual_array)

# 3. Create a 2D array using NumPy
try:
    import numpy as np
    numpy_array = np.zeros((3, 4), dtype=int)
    print("\n2D Array using NumPy:")
    print(numpy_array)
except ImportError:
    print("\nNumPy not installed, skipping NumPy example.")

# 4. Access and modify an element
print("\nAccess element at [0][2]:", array_2d[0][2])
array_2d[1][1] = 99
print("After modifying element at [1][1]:")
print(array_2d)


✅ This file covers:

Creating 2D arrays (Python & NumPy).

Accessing and modifying.

Practical matrix-like structure.


