import numpy as np

arr = np.array([[0.5, 0.2, 1.0], [0.1, 0.7, 0.3]])
indices = np.where(arr == 1.0)

coordinate = (indices[0].item(), indices[1].item())
print(coordinate)
