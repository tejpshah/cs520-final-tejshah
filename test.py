import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
std = np.std(A, axis=None, ddof=1)
print(std)
