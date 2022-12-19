import numpy as np

# Create a 2D NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 8]])

# Find the maximum value in the array
max_value = arr.max()

# Count the number of elements that have the maximum value
max_count = np.count_nonzero(np.equal(arr, max_value))

print(f'Number of maximum elements: {max_count}')
