import numpy as np

import numpy as np

def get_num_nonzero_clusters(matrix):
    """
    Find the number of connected non-zero clusters in a 2D numpy matrix.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        The 2D numpy matrix to search for non-zero clusters.
    
    Returns
    -------
    int
        The number of non-zero clusters found in the matrix.
    
    Examples
    --------
    >>> matrix = np.array([[1, 0, 4], [2, 0, 3], [0, 5, 0]])
    >>> get_num_nonzero_clusters(matrix)
    3
    """
    # Initialize a visited matrix to track which cells have been visited
    visited = np.zeros(matrix.shape, dtype=bool)
    
    def dfs(i, j):
        """
        Depth-first search helper function.
        """
        # Mark the current cell as visited
        visited[i, j] = True
        
        # Check the surrounding cells
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # If the surrounding cell is valid and has a non-zero value, 
            # and has not been visited yet, recursively search it
            if (0 <= i + di < matrix.shape[0] and 
                0 <= j + dj < matrix.shape[1] and 
                matrix[i + di, j + dj] and 
                not visited[i + di, j + dj]):
                dfs(i + di, j + dj)
    
    # Initialize the number of non-zero clusters found
    num_clusters = 0
    
    # Iterate through the matrix and search for non-zero clusters
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] and not visited[i, j]:
                # If a non-zero cell has not been visited, it belongs to a new cluster
                num_clusters += 1
                dfs(i, j)
    
    return num_clusters

matrix = np.array([[1, 0, 4], [2, 0, 3], [1, 1, 1]])
print(get_num_nonzero_clusters(matrix))