import random 
import heapq
import numpy as np 
import random 
import matplotlib.pyplot as plt 
from kmeans import get_k, get_com_com_kclusters, get_com_kclusters, avg_clustercom_to_clustercomcom

class AStarTuple():

    """THIS CLASS WILL MAKE IT EASIER TO USE HEAPS WITH ASTAR."""

    def __init__(self, totalcost, probabilities):
        self.totalcost = totalcost 
        self.probabilities = probabilities
        self.hashableprobability = tuple(self.probabilities.flatten())
    
    def __eq__(self, other) -> bool:
        return self.hashableprobability == other.hashableprobability
    
    def __lt__(self, other) -> bool:
        return self.totalcost < other.totalcost 
    
    def __gt__(self, other) -> bool:
        return self.totalcost > other.totalcost 

class Agent():

    def __init__(self, path ="reactors/Thor23-SA74-VERW-Schematic (Classified).txt"):

        # this stores the path of the nuclear reactor 
        self.path = path

        # this stores the sequence of actions taken by the agent
        self.actions = list() 

        # this represents the nuclear reactor as a 2D matrix
        self.reactor = self.init_nuclear_reactor_config()

        # this represents probability of being at a cell as a matrix
        self.probabilities = self.init_probability_matrix()

        # this initializes the set of invalid moves for a configuration
        self.invalid_moves = self.init_invalid_actions()

        # creates a visited set
        self.visited = {tuple(self.probabilities.flatten())}

        # this runs the debug command
        # self.debug()

    # "INTELLIGENT" LOGIC TO MOVE THE AGENT WITH RESPECT TO THE CORRECT SEQUENCE
    
    def entropy(self, probabilities):
        """calculates the total entropy for a probability matrix wherever probability is non-negative."""
        entropy = 0
        for i in range(probabilities.shape[0]):
            for j in range(probabilities.shape[1]):
                if probabilities[i,j] > 0:
                    entropy += -np.log(probabilities[i,j]) * probabilities[i,j]
        return entropy 

    def utility(self, new_probs):
        """
        returns the utlity of a new state
        """
        return self.entropy(new_probs) * min(0.1, self.f(new_probs)) * min(0.1, self.g(new_probs))
        #return self.entropy(new_probs) * self.get_avg_distances_between_clusters(new_probs)
    
    def information_gained(self, old_probs, new_probs):
        """
        returns the information gained from moving to a new state 
        """
        #print(f"THE ACTION HAS AN AVERAGE DISTANCE TO COM OF {avg_clustercom_to_clustercomcom(new_probs)}")
        
        information_gain = self.entropy(new_probs) - self.entropy(old_probs)
        if information_gain == 0: return self.g(new_probs)
        else: return information_gain 

    def move_nonverbose(self):
        
        # stores the values taken for each action
        qtable = dict() 

        # the discount factor to discount future rewards and next constants
        BETA = 0.90; NEXT_STATES = ["U", "D", "L", "R"]

        # iterates through all possible next states
        for command in NEXT_STATES:

            # returns the next state probabilities after transitioning after a command 
            next_state = self.transition(self.probabilities, command)

            # the reward at the current time step 
            current_reward = self.information_gained(self.probabilities, next_state)
        


            # returns the utility of transitioning to the next state 
            expected_future_reward = 0 

            # we predict the value one step into the future
            for lookahead in NEXT_STATES:

                # returns the looakahead_next_state probabilities after trainsitioning
                lookahead_next_state = self.transition(next_state, lookahead)

                # computes the utility for this forward lookahead state and adds it to sum
                expected_future_reward += self.utility(lookahead_next_state)

            # we compute the sum of the expected future reward 
            total_reward = current_reward + BETA * expected_future_reward

            # we hash the command and the assosciated total reward 
            qtable[command] = total_reward

            # if tuple(next_state.flatten()) in self.visited: qtable[command] *= 3

        # penalize values that were last up to be to have down be twice unlikely, same things with right_left

        """
        if len(self.actions) > 0 and self.actions[-1] == "U":
            qtable["D"] *= 2 
        elif len(self.actions) > 0 and self.actions[-1] == "D":
            qtable["U"] *= 2 
        elif len(self.actions) > 0 and self.actions[-1] == "L":
            qtable["R"] *= 2 
        elif len(self.actions) > 0 and self.actions[-1] == "R":
            qtable["L"] *= 2 
        """

        # after we have found all the qvalues for the actions, select the action with the min qvalue 
        possible_actions = list()    
        minimal_qvalue = min(qtable.values())
        for action, qvalue in qtable.items():
            if qvalue == minimal_qvalue: 
                possible_actions.append(action)

        action_taken = random.choice(possible_actions)
        self.actions.append(action_taken)
        self.probabilities = self.transition(self.probabilities, action_taken)
        self.visited.add(tuple(self.probabilities.flatten()))


        print(f"TOOK {len(self.actions)} actions")
        if len(self.actions) % 1000 == 0:
            self.visualize_nuclear_reactor(self.probabilities)
            self.visualize_nuclear_reactor_3d(self.probabilities)

    def move(self):
        
        # stores the values taken for each action
        qtable = dict() 

        # the discount factor to discount future rewards and next constants
        BETA = 0.90; NEXT_STATES = ["U", "D", "L", "R"]

        # iterates through all possible next states
        for command in NEXT_STATES:

            print(f"\nThe command being executed is {command}")

            # returns the next state probabilities after transitioning after a command 
            next_state = self.transition(self.probabilities, command)

            # the reward at the current time step 
            current_reward = self.information_gained(self.probabilities, next_state)
            
            
            print(f"The current reward is {current_reward}")


            # returns the utility of transitioning to the next state 
            expected_future_reward = 0 

            # we predict the value one step into the future
            for lookahead in NEXT_STATES:

                # returns the looakahead_next_state probabilities after trainsitioning
                lookahead_next_state = self.transition(next_state, lookahead)

                # computes the utility for this forward lookahead state and adds it to sum
                expected_future_reward += self.utility(lookahead_next_state)

            # we compute the sum of the expected future reward 
            total_reward = current_reward + BETA * expected_future_reward

            print(f"The total reward is {total_reward}\n")

            # we hash the command and the assosciated total reward 
            qtable[command] = total_reward

            if tuple(next_state.flatten()) in self.visited: qtable[command] *= 3

        # penalize values that were last up to be to have down be twice unlikely, same things with right_left

        """
        if len(self.actions) > 0 and self.actions[-1] == "U":
            qtable["D"] *= 2 
        elif len(self.actions) > 0 and self.actions[-1] == "D":
            qtable["U"] *= 2 
        elif len(self.actions) > 0 and self.actions[-1] == "L":
            qtable["R"] *= 2 
        elif len(self.actions) > 0 and self.actions[-1] == "R":
            qtable["L"] *= 2 
        
        """

        

        print(f"\nJust completed computation for the policy...")
        print(f"The qtable from the curent state is {qtable}")

        # after we have found all the qvalues for the actions, select the action with the min qvalue 
        possible_actions = list()    
        minimal_qvalue = min(qtable.values())
        for action, qvalue in qtable.items():
            if qvalue == minimal_qvalue: 
                possible_actions.append(action)

        action_taken = random.choice(possible_actions)
        self.actions.append(action_taken)
        self.probabilities = self.transition(self.probabilities, action_taken)
        self.visited.add(tuple(self.probabilities.flatten()))
   
        print(f"Just decided to take the next following action: {action_taken}")
        print(f"Now, my probabilities and state are updated as follows:")
        print(f"{self.probabilities}")
        print(f"I have taken {len(self.actions)} commands so far!")

        self.visualize_nuclear_reactor(self.probabilities)
        self.visualize_nuclear_reactor_3d(self.probabilities)

    def move_deterministically(self, deactivating_path):
        self.deactivating_path = deactivating_path
        commands = self.load_deactivating_sequence()
        print(commands)
        for action in commands:
            self.visualize_nuclear_reactor(self.probabilities)
            self.visualize_nuclear_reactor_3d(self.probabilities)
            self.probabilities = self.transition(self.probabilities, action)

    def f(self, matrix):
        
        def distance(p1, p2):
            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) ** 0.5
        
        # Get the list of centers of mass for each cluster
        coms = get_com_kclusters(matrix)

       #  print(coms)

        distances = list() 

        # finds the minimum distance between two cluster
        for c_i in coms:
            for c_j in coms:
                distances.append(distance(c_i, c_j))

        return sum(distances) / len(distances)

    def g(self, matrix):
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

        return max(distances)

    def a_star(self):

        print("\n ------ INITIALIZE THE A STAR ALGORITHM -----\n")

        def entropy(probabilities):
                entropy = 0
                for i in range(probabilities.shape[0]):
                    for j in range(probabilities.shape[1]):
                        if probabilities[i,j] > 0:
                            entropy += -np.log(probabilities[i,j]) * probabilities[i,j]
                return entropy 

        """
        def cost_entropy(old_p, new_p):
            return 2*entropy(new_p) - entropy(old_p)         

        def cost_basic(curr_seq):
            return len(curr_seq)

        def cost(curr_probs, next_probs):
            # @returns the cost up to the point as number of steps taken
            #return 0
            #return np.mean(next_probs) - np.mean(curr_probs)
            value = np.std(next_probs, axis=None, ddof=1) - np.std(curr_probs, axis=None, ddof=1)
            return value * self.get_num_nonzero_clusters(next_probs)
            #return 1 - (next_probs.max() - curr_probs.max())

        def heuristic(before_probs, after_probs):
            # @returns negative log likelihood of the cell with the highest probability
            
            #return entropy(after_probs) - entropy(before_probs)
            return entropy(after_probs)

            #return -np.log(probabilities.max())           """

        starting_entropy = entropy(self.probabilities)

        def h(next_probs):
            #denominator = self.num_white_cells
            #denominator = 2
            #return entropy(next_probs) * self.get_num_nonzero_clusters(next_probs)

            value = entropy(next_probs)
            value2 = avg_clustercom_to_clustercomcom(next_probs)
            value3 = get_k(next_probs)
            print("\n----------------------------\n")
            print(f"\nTHE ENTROPY OF THE NEXT PROB STATE IS: {value}")
            print(f"THE AVG CLUSTER DISTANCE OF THE NEXT PROB STATE IS: {value2}")
            print(f"THE # OF CLUSTERS ARE: {value3}\n")
            return entropy(next_probs) + avg_clustercom_to_clustercomcom(next_probs) * get_k(next_probs)
            
        def g(prev_probs, next_probs):
            return entropy(next_probs) - starting_entropy
        

        print(f"\nSTARTING THE A STAR ALGORITHM...")

        # initialize the heap list and visited set
        heap, visited = list(), set() 

        # initialize heap with ( cost(s), ( s, seq(s) ) )
        s0 = AStarTuple(0 + h(self.probabilities), self.probabilities)
        heap.append( (s0, []) )

        print(f"\nInitialized The Heap:")
        print(heap)

        # run the A* algorithm until termination 
        while len(heap) > 0:

            print("\n ------ POP OFF MIN ITEM FROM HEAP -----\n")

            # retrieves the item of minimal cost 
            curr_state, curr_seq = heapq.heappop(heap)

            print(f"\nPopped off minimal cost item from heap!")
            print(f"The current sequence is: {curr_seq}")
            print(f"The probabilities are:")
            print(curr_state.probabilities)

            # self.visualize_nuclear_reactor(curr_state.probabilities)
            # self.visualize_nuclear_reactor_3d(curr_state.probabilities)

            # returns the sequence of moves if terminal state
            if self.is_terminal_state(curr_state.probabilities):
                print(f"\nWe reached a terminal state!")
                self.actions = curr_seq 
                return curr_seq  
            
            # mark the current state as visited 
            visited.add(curr_state.hashableprobability)

            # print(f"\n The current visited set is: {visited}")

            print("\n ------ LOOKING AT TRANSITIONS NOW -----\n")

            # iterate through every possible action possible 
            for action in ["U", "D", "L", "R"]:

                next_probs = self.transition(curr_state.probabilities, action)

                # if we've already visited this state before continue 
                if tuple(next_probs.flatten()) not in visited: 

                    #print(f"\nIf we move with action {action}, we get the new proabilities:")
                    #print(next_probs)


                    #total_cost = round(heuristic(next_probs) + cost(curr_seq), 1)
                    #next_seq = curr_seq + [action]

                    #total_cost, next_seq= heuristic(next_probs) + cost(curr_state.probabilities, next_probs), curr_seq + [action]

                    #total_cost, next_seq= curr_state.totalcost + heuristic(curr_state.probabilities, next_probs) , curr_seq + [action]

                    #total_cost, next_seq= cost_entropy(curr_state.probabilities, next_probs) , curr_seq + [action]
                    total_cost, next_seq= g(curr_state.probabilities, next_probs) + h(next_probs), curr_seq + [action]

                    s1 = AStarTuple(total_cost, next_probs)

                    if len(heap) < 1000:
                        heapq.heappush(heap, (s1, next_seq))
                    else: 
                        heapq.heappushpop(heap, (s1, next_seq))

                    print(f"\nWe are pushing this information to the heap...")
                    print(f"The total cost: {total_cost}")
                    print(f"The next sequence: {next_seq}")
                   #print(f"The new probabilities:")
                    #print(s1.probabilities)

        return heap 

    def get_num_nonzero_clusters(self, matrix):
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
  
    def avg_clustercom_to_clustercomcom_bfs(self, matrix):
        
        # find the center of mass of the center of mass
        comcom = get_com_com_kclusters(matrix)

        # gets the list of centers of mass for each cluster
        coms = get_com_kclusters(matrix)

        # a matrix of distances 
        distances = list() 

        # go through every com in the cluster
        for com in coms:
            p1 = self.snap_tuple_to_2d_grid(self.reactor, com)
            p2 = self.snap_tuple_to_2d_grid(self.reactor, comcom)
            distances.append(len(self.shortest_path(self.reactor, p1, p2)))

        average = sum(distances) / len(distances) 

        return max(distances)

    def snap_tuple_to_2d_grid(self, maze, tuple_coordinates):
        # Convert the input maze to a NumPy array
        maze = np.array(maze)
        
        # Find the coordinates of all the zeros in the maze
        zero_coords = np.argwhere(maze == 0)
        
        # Calculate the distances between the input tuple and all the zero coordinates
        distances = np.linalg.norm(zero_coords - tuple_coordinates, axis=1)
        
        # Find the index of the zero coordinate that is closest to the input tuple
        min_index = np.argmin(distances)
        
        # Return the coordinates of the closest zero
        return tuple(zero_coords[min_index])

    def shortest_path(self, maze, start, end):
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

    def get_two_closest_cluster_distances(self, matrix):
        
        def distance(p1, p2):
            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) ** 0.5
        
        # Get the list of centers of mass for each cluster
        coms = get_com_kclusters(matrix)

        print(self.reactor)

        # finds the minimum distance between two cluster
        min_distance = float("inf")
        for c_i in coms:
            for c_j in coms:
                d = distance(c_i, c_j)
                if c_i != c_j and d < min_distance:
                    min_distance = d
        return min_distance

    def get_avg_distances_between_clusters(self, matrix):
        
        def distance(p1, p2):
            return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 ) ** 0.5
        
        # Get the list of centers of mass for each cluster
        coms = get_com_kclusters(matrix)

       #  print(coms)

        distances = list() 

        # finds the minimum distance between two cluster
        for c_i in coms:
            for c_j in coms:
                distances.append(distance(c_i, c_j))
        
        return sum(distances) / len(distances)

    # INITIALIZATION FUNCTIONS FOR NUCLEAR REACTOR, PROBABILITIES, & INVALID ACTIONS
    
    def init_nuclear_reactor_config(self):
        """
        generates a 2D numpy matrix that represents the relevant configuration of a nuclear reactor. 
        @returns reactor : 2D numpy matrix with the configuration of the reactor {"_" -> 0, "X" -> 1}
        """
        # reads in the file as a list of strings
        with open(self.path, "r") as f:
            lines = f.readlines() 
        
        # remove leading/trailing white spaces in the text file
        lines = [list(line.strip()) for line in lines]

        # {"_" -> 0, "X" -> 1} conversion for the numpy matrix 
        lines = [[1 if x == "X" else 0 for x in line] for line in lines]

        # convert the list of lists to a 2D numpy array 
        reactor = np.array(lines)

        # return a 2D numpy array of the nuclear reactor configuration
        return reactor 
    
    def init_probability_matrix(self):
        """
        generates initial transition matrix for where the drone can be as 1 / [# of white cells (aka reactor with 0)] in each location of matrix
        @returns probabilities : represents the probability of drone being at cell (i,j). 
        """
        # count the number of white cells in the nuclear reactor
        num_white_cells = (self.reactor == 0).sum()

        # stores the number of white cells
        self.num_white_cells = num_white_cells

        print(f"The number of white cells are {num_white_cells}.")

        # initial probability matrix of same shape as reactor with all being equally likely
        probabilities = np.ones(self.reactor.shape) * (1 / num_white_cells)

        # set the probability of black cells in reactor to 0
        probabilities[self.reactor == 1] = 0 

        # return the probability matrix
        return probabilities 
    
    def init_invalid_actions(self):
        """
        whenever an agent moves from a cell to another cell we have to check whether the move is invalid.
        if the move is invalid, then the agent must keep the current location and "not" move in specified direction.
        @return invalid_moves : dictionary of tuple coordinates (i,j) that are invalid moves for the agent. 
        """
        # add all invalid moves that correspond to blocked cells in the grid
        invalid_moves = {(index[0], index[1]) for index in np.argwhere(self.reactor == 1)}

        # add all invalid moves that are out of bounds 
        for i in range(-1, self.reactor.shape[0] + 1):
            for j in range(-1, self.reactor.shape[1] + 1):
                if i < 0 or j < 0 or i > (self.reactor.shape[0]-1) or j > (self.reactor.shape[1]-1):
                    invalid_moves.add((i,j))
        
        # return the invalid moves set
        return invalid_moves

    # FUNCTIONALITY TO ENABLE THE AGENT TO MOVE AND UPDATE PROBABILITIES BASED ON ACTION TAKEN

    def transition(self, probabilities, action):
        """
        @returns next probability matrix based on the command executed
        """
        if action == "R":
            return self.move_right(probabilities)
        elif action == "L":
            return self.move_left(probabilities)
        elif action == "U":
            return self.move_up(probabilities) 
        elif action == "D":
            return self.move_down(probabilities)

    def move_down(self, probabilities):
        """
        this updates the probabilities matrix if the agent moves down. (i,j) -> (i+1,j)
        """
        p_down = np.zeros(probabilities.shape)
        for i in range(0, self.reactor.shape[0]):
            for j in range(0, self.reactor.shape[1]):
                if (i+1, j) not in self.invalid_moves:
                    p_down[i+1, j] += probabilities[i,j]
                else: p_down[i, j] += probabilities[i,j]
        return p_down 

    def move_up(self, probabilities):
        """
        this updates the probabilities matrix if the agent moves down. (i,j) -> (i-1,j)
        """
        p_up = np.zeros(probabilities.shape)
        for i in range(0, self.reactor.shape[0]):
            for j in range(0, self.reactor.shape[1]):
                if (i-1, j) not in self.invalid_moves:
                    p_up[i-1, j] += probabilities[i,j]
                else: p_up[i, j] += probabilities[i,j]
        return p_up 
    
    def move_left(self, probabilities):
        """
        this updates the probabilities matrix if the agent moves down. (i,j) -> (i,j-1)
        """
        p_left = np.zeros(probabilities.shape)
        for i in range(0, self.reactor.shape[0]):
            for j in range(0, self.reactor.shape[1]):
                if (i, j-1) not in self.invalid_moves:
                    p_left[i, j-1] += probabilities[i,j]
                else: p_left[i, j] += probabilities[i,j]
        return p_left 
    
    def move_right(self, probabilities):
        """
        this updates the probabilities matrix if the agent moves down. (i,j) -> (i,j+1)
        """
        p_left = np.zeros(probabilities.shape)
        for i in range(0, self.reactor.shape[0]):
            for j in range(0, self.reactor.shape[1]):
                if (i, j+1) not in self.invalid_moves:
                    p_left[i, j+1] += probabilities[i,j]
                else: p_left[i, j] += probabilities[i,j]
        return p_left 

    def is_terminal_state(self, probabilities):
        """
        ends the game if 1.0 is in any of the probabilities (i.e. we are 100% confident on localizing the drone)
        """ 
        for i in range(0, probabilities.shape[0]):
            for j in range(0, probabilities.shape[1]):
                value = probabilities[i,j]
                if value > 0.999: 
                    return True 
        return False 

    def get_localized_robot_location(self):
        """
        find the cell where there is a 100% probability of containing the robot. 
        """
        indices = np.where(self.probabilities == 1.0)
        coordinate = (indices[0].item(), indices[1].item())
        print(f"THE ROBOT IS 100% LOCALIZED TO BE AT {coordinate}")
        return coordinate 

    # DEBUGGING FUNCTIONS FOR PRINTING OUTPUT TO TERMINAL AND VISUALIZING GAME STATE AND OTHER UTILITIES

    def visualize_nuclear_reactor(self, probabilities):
        """
        generates a visualization of the nuclear reactor configuration along with probs of being at a cell. 0 -> white, 1 -> black
        """
        # set the colormap and color limits 
        plt.imshow(probabilities, cmap='magma', vmin=probabilities.min(), vmax=probabilities.max())

        # show the color bar
        plt.colorbar()

        # removes the tick marks and labels
        plt.gca().tick_params(which="both", length=0)
        plt.gca().set_xticklabels([])
        plt.gca().set_yticklabels([])

        # sets the tick labels at center of each cell to be the values in each cell
        for i in range(self.reactor.shape[0]):
            for j in range(self.reactor.shape[1]):
                if self.reactor[i,j] != 1:
                    plt.text(j, i, round(probabilities[i,j], 3), ha="center", va="center", color="blue", fontsize=6)

        # visualizes the nuclear reactor
        plt.show()

    def visualize_nuclear_reactor_3d(self, probabilities):
        """
        generates a 3D visualization of the nuclear reactor configuration with probs as the height above the surface.
        """

        # create a figure and 3D Axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # create a 2D array of x and y values for the 3D plot
        x, y = np.meshgrid(range(self.reactor.shape[1]), range(self.reactor.shape[0]))

        # plot the surface using the probabilities as the z values
        ax.plot_surface(x, y, probabilities, cmap='magma',  vmin=probabilities.min(), vmax=probabilities.max())

        # show the plot
        plt.show()

    def load_deactivating_sequence(self):
        """
        this will load into an array the sequence of commands. 
        """
        # initialize the command sequence
        command_sequence = list()

        # opens the command sequence
        with open(self.deactivating_path, 'r') as file:
            commands = file.read()

        # iterate through the contents of the file and append each character to the array
        for move in commands:
            command_sequence.append(move)
        
        # returns the list of commands in the command sequence
        return [command for command in command_sequence if command != ","]

    def debug(self):
        """
        prints out to terminal the probabilities, nuclear reactor, and other debugging information.
        """
        
        print(f"\nTHE REACTOR IS:")
        print(self.reactor)

        print(f"\nTHE PROBABITIES ARE:")
        print(self.probabilities)

        self.visualize_nuclear_reactor()
        self.visualize_nuclear_reactor_3d()

        print(f"\nMOVE PROBABILITIES NOW:")
        self.probabilities = self.move_down()
        print(self.probabilities)

        self.visualize_nuclear_reactor()
        self.visualize_nuclear_reactor_3d()

if __name__ == "__main__":
    agent = Agent(path="reactors/toyexample3.txt")
    #agent = Agent()
    #agent.a_star()

    while not agent.is_terminal_state(agent.probabilities):
        agent.move()
    print(f"The optimal action sequence is of length {len(agent.actions)} is {agent.actions}!")

    #agent.move_deterministically(deactivating_path="sequences/sequence-toyexample3.txt")
    # agent = Agent()
