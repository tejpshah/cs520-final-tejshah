import numpy as np

# Create a 2D NumPy array with some elements equal to 1
arr = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

# Find the indices of all elements with the value 1
indices = np.argwhere(arr == 1)

# Create a dictionary to store the tuples
coordinates = {}

# Iterate over the indices and create a tuple for each index
for index in indices:
    row = index[0]
    col = index[1]
    coordinates[(row, col)] = (row, col)

print(coordinates)
