import numpy as np 

def get_k(matrix):
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
  
def get_k_clusters(matrix):
    """
    Find all the connected non-zero clusters in a 2D numpy matrix.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        The 2D numpy matrix to search for non-zero clusters.
    
    Returns
    -------
    list of list of tuple
        A list of non-zero clusters, where each cluster is represented as a list of 2-tuples of the form (row, column).
    
    Examples
    --------
    >>> matrix = np.array([[1, 0, 4], [2, 0, 3], [0, 5, 0]])
    >>> get_k_clusters(matrix)
    [[(0, 0), (1, 0)], [(0, 2), (1, 2)], [(2, 1)]]
    """
    # Initialize a visited matrix to track which cells have been visited
    visited = np.zeros(matrix.shape, dtype=bool)
    
    # Initialize a list to store the clusters
    clusters = []
    
    def dfs(i, j):
        """
        Depth-first search helper function.
        """
        # Mark the current cell as visited
        visited[i, j] = True
        
        # Add the current cell to the current cluster
        clusters[-1].append((i, j))
        
        # Check the surrounding cells
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            # If the surrounding cell is valid and has a non-zero value, 
            # and has not been visited yet, recursively search it
            if (0 <= i + di < matrix.shape[0] and 
                0 <= j + dj < matrix.shape[1] and 
                matrix[i + di, j + dj] and 
                not visited[i + di, j + dj]):
                dfs(i + di, j + dj)
    
    # Iterate through the matrix and search for non-zero clusters
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] and not visited[i, j]:
                # If a non-zero cell has not been visited, it belongs to a new cluster
                clusters.append([])
                dfs(i, j)
    
    return clusters

def get_weighted_k_clusters(matrix):
    """
    Find all the connected non-zero clusters in a 2D numpy matrix, and calculate the weighted value of each cluster.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        The 2D numpy matrix to search for non-zero clusters.
    
    Returns
    -------
    list of list of tuple
        A list of non-zero clusters, where each cluster is represented as a list of 2-tuples of the form (row * value, column * value).
    
    Examples
    --------
    >>> matrix = np.array([[1, 0, 4], [2, 0, 3], [0, 5, 0]])
    >>> get_weighted_k_clusters(matrix)
    [[(0, 0), (2, 0)], [(0, 8), (3, 6)], [(10, 5)]]    
    """
    # Find all the non-zero clusters in the matrix
    clusters = get_k_clusters(matrix)
    
    # Initialize a list to store the weighted clusters
    weighted_clusters = []
    
    # Iterate over each cluster
    for cluster in clusters:
        # Initialize a list to store the weighted values for the current cluster
        weighted_cluster = []
        
        # Iterate over each cell in the cluster
        for i, j in cluster:
            # Calculate the weighted value for the cell by multiplying the row and column indices with the value at that cell in the matrix
            weighted_cluster.append((i * (matrix[i, j]), j * (matrix[i, j])))
        
        # Add the weighted cluster to the list of weighted clusters
        weighted_clusters.append(weighted_cluster)
    
    return weighted_clusters

def get_com_kclusters(matrix):
    """
    Calculate the center of mass for each cluster in a 2D numpy matrix.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        The 2D numpy matrix to search for non-zero clusters.
    
    Returns
    -------
    list of tuple
        A list of 2-tuples of the form (average x, average y) representing the center of mass for each cluster.
    
    Examples
    --------
    >>> matrix = np.array([[1, 0, 4], [2, 0, 3], [0, 5, 0]])
    >>> get_com_kclusters(matrix)
        [(1.0, 0.0), (1.5, 7.0), (10.0, 5.0)]
    """
    # Get the list of weighted clusters
    clusters = get_weighted_k_clusters(matrix)
    
    # Initialize a list to store the center of mass for each cluster
    coms = []
    
    # Iterate through each cluster
    for cluster in clusters:
        # Initialize the sums for the x and y values
        x_sum = 0
        y_sum = 0
        
        # Iterate through each point in the cluster
        for point in cluster:
            # Add the x and y values for the point to the sums
            x_sum += point[0]
            y_sum += point[1]
        
        # Calculate the average x and y values for the cluster
        x_avg = x_sum / len(cluster)
        y_avg = y_sum / len(cluster)
        
        # Add the center of mass for the cluster to the list
        coms.append((x_avg, y_avg))
    
    # Return the list of centers of mass
    return coms

def get_com_com_kclusters(matrix):
    """
    Calculate the center of mass for the center of mass of all the clusters in a 2D numpy matrix.
    
    Parameters
    ----------
    matrix : numpy.ndarray
        The 2D numpy matrix to search for non-zero clusters.
    
    Returns
    -------
    tuple
        A 2-tuple of the form (average x, average y) representing the center of mass for the center of mass of all the clusters.
    
    Examples
    --------
    >>> matrix = np.array([[1, 0, 4], [2, 0, 3], [0, 5, 0]])
    >>> get_com_com_kclusters(matrix)
        (4.166666666666667, 4.0)
    """
    # Get the list of centers of mass for each cluster
    coms = get_com_kclusters(matrix)
    
    # Initialize the sums for the x and y values
    x_sum = 0
    y_sum = 0
    
    # Iterate through each center of mass
    for com in coms:
        # Add the x and y values for the center of mass to the sums
        x_sum += com[0]
        y_sum += com[1]
    
    # Calculate the average x and y values for all the centers of mass
    x_avg = x_sum / len(coms)
    y_avg = y_sum / len(coms)
    
    # Return the center of mass for all the centers of mass
    return (x_avg, y_avg)

def avg_clustercom_to_clustercomcom(matrix):
    
    # find the center of mass of the center of mass
    comcom = get_com_com_kclusters(matrix)

    # gets the list of centers of mass for each cluster
    coms = get_com_kclusters(matrix)

    # a matrix of distances 
    distances = list() 

    # go through every com in the cluster
    for com in coms:
        x, y = com[0], com[1] 
        a, b = comcom[0], comcom[1] 
        distances.append( ( (x-a)**2 + (y-b)**2 ) ** 0.5  )

    average = sum(distances) / len(distances) 

    return max(distances)

if __name__ == "__main__":
    matrix = np.array([[1, 0, 4], [2, 0, 3], [0, 5, 0]])
    print(get_k(matrix))
    print(get_k_clusters(matrix))
    print(get_weighted_k_clusters(matrix))
    print(get_com_kclusters(matrix))
    print(get_com_com_kclusters(matrix))
    print(avg_clustercom_to_clustercomcom(matrix))