import numpy as np 

################# HELPER FUNCTIONS FOR CLUSTERING ALGORITHMS #################

def get_k(matrix):
    """
    this will return the number of islands that have non-zero connected components in a matrix.
    this means that we will get the number of probability mass clusters for our particular matrix. 
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

def get_k_clusters_points(matrix):
    """
    this will return the number of islands and the values for those that have non-zero connected components in a matrix.
    this means that we will get the indexes consisting of each probability mass clusters for our particular matrix. 
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

def get_k_clusters_centroids(matrix):
    """
    this will return the centroid of each probabillity mass cluster in the matrix. 
    the function includes coms from deprecated methods that took into account probabilities. 
    it was found that the centroids are better than the center of mass but kept all the variable names. 
    """
    # get the list of alweighted clusters
    clusters = get_k_clusters_points(matrix)
    
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

def get_k_clusters_centroid_of_centroids(matrix):
    """
    this will return the centroid of the centroid of each probabillity mass cluster in the matrix. 
    the function includes coms from deprecated methods that took into account probabilities. 
    it was found that the centroids are better than the center of mass but kept all the variable names. 
    """
    # Get the list of centers of mass for each cluster
    coms = get_k_clusters_centroids(matrix)
    
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

def snap_tuple_to_2d_grid(maze, tuple_coordinates):
  """this will help us move towards the closest coordinates"""
  x, y = tuple_coordinates
  x_idx = np.argmin(np.abs(np.arange(maze.shape[0]) - x))
  y_idx = np.argmin(np.abs(np.arange(maze.shape[1]) - y))
  return x_idx, y_idx

def bfs(maze, start, end):
    """this will compute bfs on two points on a grid"""
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
    return len(path[::-1])

################# OBJECTIVE FUNCTIONS FOR MULTI-OBJECTIVE RL #################

def calculate_entropy(probabilities):
    """
    this function calculates the entropy for all probabilities in matrix >0 
    """
    entropy = 0
    for i in range(probabilities.shape[0]):
        for j in range(probabilities.shape[1]):
            if probabilities[i,j] > 0:
                entropy += -np.log(probabilities[i,j]) * probabilities[i,j]
    return entropy 

def calculate_information_gain(old_probs, new_probs):
    """
    this function calculates the information gain as the change in entropy 
    """
    return calculate_entropy(new_probs) - calculate_entropy(old_probs)

def get_avg_pairwise_distance_cluster_centroids(matrix):
    """
    this will find the average pairwise distance between each cluster's centroid
    """
    def distance(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) ** 0.5
    
    # Get the list of centers of mass for each cluster
    coms = get_k_clusters_centroids(matrix)

    #  print(coms)

    distances = list() 

    # finds the minimum distance between two cluster
    for c_i in coms:
        for c_j in coms:
            distances.append(distance(c_i, c_j))

    return sum(distances) / len(distances)

def get_max_dist_from_kth_cluster_to_centroid_of_centroids(matrix):
    """get_max_dist_from_kth_cluster_to_centroid_of_centroids"""
    # find the center of mass of the center of mass
    comcom = get_k_clusters_centroid_of_centroids(matrix)

    # gets the list of centers of mass for each cluster
    coms = get_k_clusters_centroid_of_centroids(matrix)

    # a matrix of distances 
    distances = list() 

    # go through every com in the cluster
    for com in coms:
        x, y = com[0], com[1] 
        a, b = comcom[0], comcom[1] 
        distances.append( ( (x-a)**2 + (y-b)**2 ) ** 0.5  )

    return max(distances)

def get_two_farthest_cluster_distances(matrix):
    """start to move the two closest clusters towards each """
    
    def distance(p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) ** 0.5
    
    # Get the list of centers of mass for each cluster
    coms = get_k_clusters_centroids(matrix)

    # finds the  distance between two cluster
    max_distance = -float("inf")
    for c_i in coms:
        for c_j in coms:
            d = distance(c_i, c_j)
            if c_i != c_j and d > max_distance:
                max_distance = d

    return max_distance

def compute_utility(new_probs):
    """this will compute the utility for a particular state"""
    objective_h = calculate_entropy(new_probs)
    objective_f = min(0.1, get_avg_pairwise_distance_cluster_centroids(new_probs))
    objective_g = min(0.1, get_max_dist_from_kth_cluster_to_centroid_of_centroids(new_probs))
    return objective_h * objective_f * objective_g

def compute_reward(old_probs, new_probs):
    """this will compute the reward for a particular state"""
    objective_i = calculate_information_gain(old_probs, new_probs)
    objective_h = calculate_entropy(new_probs)
    objective_f = min(0.1, get_avg_pairwise_distance_cluster_centroids(new_probs))
    objective_g = min(0.1, get_max_dist_from_kth_cluster_to_centroid_of_centroids(new_probs))
    objective_l = get_two_farthest_cluster_distances(new_probs)
    k_clusters = get_k(new_probs)

    if k_clusters == 1:
        return calculate_entropy(new_probs)
    elif objective_i > 0:
        return objective_i * objective_h * objective_f * objective_g
    else:
        return objective_l

