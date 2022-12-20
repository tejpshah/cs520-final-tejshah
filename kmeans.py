import numpy as np 
from collections import deque

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
    #clusters = get_weighted_k_clusters(matrix)
    clusters = get_k_clusters(matrix)
    
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

def snap_tuple_to_2d_grid(maze, tuple_coordinates):
  x, y = tuple_coordinates
  x_idx = np.argmin(np.abs(np.arange(maze.shape[0]) - x))
  y_idx = np.argmin(np.abs(np.arange(maze.shape[1]) - y))
  return x_idx, y_idx

def shortest_path(maze, start, end):
    # initialize the queue with the start position
    queue = [start]
    # initialize a set to store visited positions
    visited = set()

    # initialize a dictionary to store the predecessor of each position
    predecessor = {}

    # initialize the distance of the start position to be 0
    distance = {start: 0}

    # while there are still positions in the queue
    while queue:
        # get the first position in the queue
        current_position = queue.pop(0)

        # if the current position is the end position, we are done
        if current_position == end:
            break

        # get the coordinates of the current position
        x, y = current_position

        # check the positions to the north, south, east, and west of the current position
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            # compute the coordinates of the new position
            new_x, new_y = x + dx, y + dy

            # skip the new position if it is outside the bounds of the maze
            if not (0 <= new_x < maze.shape[0] and 0 <= new_y < maze.shape[1]):
                continue

            # skip the new position if it is blocked
            if maze[new_x, new_y] == 1:
                continue

            # skip the new position if it has already been visited
            if (new_x, new_y) in visited:
                continue

            # add the new position to the queue and mark it as visited
            queue.append((new_x, new_y))
            visited.add((new_x, new_y))

            # set the distance of the new position to be the distance of the current position plus 1
            distance[(new_x, new_y)] = distance[current_position] + 1

            # set the predecessor of the new position to be the current position
            predecessor[(new_x, new_y)] = current_position

    # if the end position was not reached, return None
    if end not in distance:
        return None

    # initialize the shortest path with the end position
    path = [end]

    # use the predecessor dictionary to reconstruct the shortest path
    while path[-1] != start:
        path.append(predecessor[path[-1]])

    # return the shortest path in reverse order
    return path[::-1]

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
    # return average 
    return max(distances)

if __name__ == "__main__":
    matrix = np.array([[1, 0, 4], [2, 0, 3], [0, 5, 0]])
    print(get_k(matrix))
    print(get_k_clusters(matrix))
    print(get_weighted_k_clusters(matrix))
    print(get_com_kclusters(matrix))
    print(get_com_com_kclusters(matrix))
    print(avg_clustercom_to_clustercomcom(matrix))

    print(f"TEST")
    print(get_two_closest_cluster_distances(matrix))

    maze = np.array([[0, 1, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 1],
                    [0, 1, 0, 0]])

    tuple_coordinates = (2.2, 3.3)
    closest_coordinates = snap_tuple_to_2d_grid(maze, tuple_coordinates)
    print(closest_coordinates)  # Output: (1, 2)

    maze = np.array([
        [0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 0]
    ])

    # define the start and end positions
    start = (0, 0)
    end = (4, 4)

    # find the shortest path
    path = shortest_path(maze, start, end)

    # print the shortest path
    print(path)
