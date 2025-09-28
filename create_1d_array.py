📘 Notes in Easy Words

1D Array → A list of numbers in NumPy.
Think of it as a row of numbers: [1, 2, 3, 4]
Shape will be (n,) where n = number of elements

Why NumPy?
NumPy arrays are faster and more efficient than Python lists.
They support mathematical operations directly on arrays.

How to create?
Use np.array(list) to create a 1D array from a Python list.
Use np.arange() to create a range of numbers.
Use np.linspace() to create evenly spaced numbers.


Python Code – 1D Array
import numpy as np

# 1. Create 1D array from a Python list
arr1 = np.array([1, 2, 3, 4, 5])
print("1D array from list:", arr1)
print("Shape:", arr1.shape)
print("Data type:", arr1.dtype)

# 2. Create 1D array using arange (like range())
arr2 = np.arange(0, 10, 2)  # start=0, stop=10, step=2
print("\n1D array using arange:", arr2)

# 3. Create 1D array using linspace (evenly spaced numbers)
arr3 = np.linspace(0, 1, 5)  # 5 numbers from 0 to 1
print("\n1D array using linspace:", arr3)

Output:

1D array from list: [1 2 3 4 5]
Shape: (5,)
Data type: int64

1D array using arange: [0 2 4 6 8]

1D array using linspace: [0.   0.25 0.5  0.75 1.  ]

Real-Life Example
Suppose you track daily sales of your shop for a week: [100, 120, 90, 150, 130, 110, 160]
You can store this as a 1D NumPy array.
Then you can calculate total sales, average, or find max/min sales quickly using NumPy.
