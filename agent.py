import random 
import numpy as np 
import matplotlib.pyplot as plt 

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

        # this runs the debug command
        # self.debug()

    # "INTELLIGENT" LOGIC TO MOVE THE AGENT WITH RESPECT TO THE CORRECT SEQUENCE
    
    def move_intelligently(self):
        """
        our goal is to move the probability mass to a single point with 100% probability.
        """
        # forward lookahead to the probability matrix after taking one action 
        p_d, p_u, p_l, p_r = self.move_down(), self.move_up(), self.move_left(), self.move_right()

        # computes the number of clusters that each of the new transition probablities will generate
        c_d, c_u, c_l, c_r = self.get_num_nonzero_clusters(p_d),  self.get_num_nonzero_clusters(p_u),  self.get_num_nonzero_clusters(p_l),  self.get_num_nonzero_clusters(p_r)

        # compute a metric that "scores" the utility of the probability metric
        s_d, s_u, s_l, s_r = self.get_std(p_d), self.get_std(p_u), self.get_std(p_l), self.get_std(p_r)

        # filters the possible actions that you can take that have the minimal number of non-zero clusters
        clusters = {"D" : c_d, "U" : c_u, "R" : c_r, "L" : c_l}
        min_clusters = min(clusters.values())

        # find all possible actions that have the minimal number of clusters
        possible_actions = list() 
        for command, cluster in clusters.items():
            if cluster == min_clusters: possible_actions.append(command)

        # finds the standard deviation of all possible actions with min clusters
        utilities = dict() 
        for action in possible_actions:
            if action == "D": 
                utilities[action] = s_d 
            elif action == "L": 
                utilities[action] = s_l 
            elif action == "R": 
                utilities[action] = s_r 
            elif action == "U": 
                utilities[action] = s_u 

        # greedily select the next action that optimizes the objective function
        max_utility = max(utilities.values())

        # find all possible actions that have the maximal utility
        possible_actions = list() 
        for command, utility in utilities.items():
            if utility == max_utility: possible_actions.append(command)

        # randomly select a possible actionto avoid getting stuck in local minima
        action = random.choice(possible_actions)

        # add the action selected to the trajectory 
        self.actions.append(action)

        # select an action to take for the probabilities 
        if action == "D": 
            self.probabilities = p_d 
        elif action == "U":
            self.probabilities = p_u 
        elif action == "L":
            self.probabilities = p_l 
        elif action == "R":
            self.probabilities = p_r 

    def move_intelligently_debug(self):
        """
        our goal is to move the probability mass to a single point with 100% probability.
        """
        # forward lookahead to the probability matrix after taking one action 
        p_d, p_u, p_l, p_r = self.move_down(), self.move_up(), self.move_left(), self.move_right()

        # computes the number of clusters that each of the new transition probablities will generate
        c_d, c_u, c_l, c_r = self.get_num_nonzero_clusters(p_d),  self.get_num_nonzero_clusters(p_u),  self.get_num_nonzero_clusters(p_l),  self.get_num_nonzero_clusters(p_r)

        # compute a metric that "scores" the utility of the probability metric
        s_d, s_u, s_l, s_r = self.get_std(p_d), self.get_std(p_u), self.get_std(p_l), self.get_std(p_r)

        print(f"\nCommand: R \t c_r = {c_r} \t s_r = {s_r}")
        print(p_r)

        print(f"\nCommand: L \t c_l = {c_l} \t s_l = {s_l}")
        print(p_l)

        print(f"\nCommand: U \t c_u = {c_u} \t s_u = {s_u}")
        print(p_u)

        print(f"\nCommand: D \t c_d = {c_d} \t s_d = {s_d}")
        print(p_d)

        # filters the possible actions that you can take that have the minimal number of non-zero clusters
        clusters = {"D" : c_d, "U" : c_u, "R" : c_r, "L" : c_l}
        min_clusters = min(clusters.values())

        # find all possible actions that have the minimal number of clusters
        possible_actions = list() 
        for command, cluster in clusters.items():
            if cluster == min_clusters: possible_actions.append(command)
    
        print(f"\nPossible actions filtered by cluster: {possible_actions}")

        # finds the standard deviation of all possible actions with min clusters
        utilities = dict() 
        for action in possible_actions:
            if action == "D": 
                utilities[action] = s_d 
            elif action == "L": 
                utilities[action] = s_l 
            elif action == "R": 
                utilities[action] = s_r 
            elif action == "U": 
                utilities[action] = s_u 

        # greedily select the next action that optimizes the objective function
        max_utility = max(utilities.values())

        # find all possible actions that have the maximal utility
        possible_actions = list() 
        for command, utility in utilities.items():
            if utility == max_utility: possible_actions.append(command)
        
        print(f"\nPossible actions filtered now by utilities: {possible_actions}")

        # randomly select a possible actionto avoid getting stuck in local minima
        action = random.choice(possible_actions)
        print(f"\nAction Taken: {action}")

        # add the action selected to the trajectory 
        self.actions.append(action)

        # select an action to take for the probabilities 
        if action == "D": 
            self.probabilities = p_d 
        elif action == "U":
            self.probabilities = p_u 
        elif action == "L":
            self.probabilities = p_l 
        elif action == "R":
            self.probabilities = p_r 
        
        self.visualize_nuclear_reactor()
        self.visualize_nuclear_reactor_3d()

    def move_deterministically(self, deactivating_path="sequences/sequence-toyexample2.txt"):
        """
        runs the AI agent towards a specified sequence of commands. 
        """

        # loads in a sequence of actions to localize the robot 
        self.deactivating_path = deactivating_path
        command_sequence = self.load_deactivating_sequence()

        print(command_sequence)

        # runs a sequence of commands 
        for command in command_sequence:

            print(f"MOVING {command}!")

            # update probabilities according to command 
            if command == "L": 
                self.probabilities = self.move_left()
            elif command == "R": 
                self.probabilities = self.move_right()
            elif command == "U": 
                self.probabilities = self.move_up()
            elif command == "D": 
                self.probabilities = self.move_down()

            # visualize and 2D and 3D probability plots
            self.visualize_nuclear_reactor()
            self.visualize_nuclear_reactor_3d()
        
        # print out the location of localized coordinate
        self.get_localized_robot_location()
        
    def get_std(self, pmatrix):
        """
        returns the sample standard deviation of a matrix to determine which location to move to
        """
        return np.std(pmatrix, axis=None, ddof=1) 

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

    def move_down(self):
        """
        this updates the probabilities matrix if the agent moves down. (i,j) -> (i+1,j)
        """
        p_down = np.zeros(self.probabilities.shape)
        for i in range(0, self.reactor.shape[0]):
            for j in range(0, self.reactor.shape[1]):
                if (i+1, j) not in self.invalid_moves:
                    p_down[i+1, j] += self.probabilities[i,j]
                else: p_down[i, j] += self.probabilities[i,j]
        return p_down 

    def move_up(self):
        """
        this updates the probabilities matrix if the agent moves down. (i,j) -> (i-1,j)
        """
        p_up = np.zeros(self.probabilities.shape)
        for i in range(0, self.reactor.shape[0]):
            for j in range(0, self.reactor.shape[1]):
                if (i-1, j) not in self.invalid_moves:
                    p_up[i-1, j] += self.probabilities[i,j]
                else: p_up[i, j] += self.probabilities[i,j]
        return p_up 
    
    def move_left(self):
        """
        this updates the probabilities matrix if the agent moves down. (i,j) -> (i,j-1)
        """
        p_left = np.zeros(self.probabilities.shape)
        for i in range(0, self.reactor.shape[0]):
            for j in range(0, self.reactor.shape[1]):
                if (i, j-1) not in self.invalid_moves:
                    p_left[i, j-1] += self.probabilities[i,j]
                else: p_left[i, j] += self.probabilities[i,j]
        return p_left 
    
    def move_right(self):
        """
        this updates the probabilities matrix if the agent moves down. (i,j) -> (i,j+1)
        """
        p_left = np.zeros(self.probabilities.shape)
        for i in range(0, self.reactor.shape[0]):
            for j in range(0, self.reactor.shape[1]):
                if (i, j+1) not in self.invalid_moves:
                    p_left[i, j+1] += self.probabilities[i,j]
                else: p_left[i, j] += self.probabilities[i,j]
        return p_left 

    def is_terminal_state(self):
        """
        ends the game if 1.0 is in any of the probabilities (i.e. we are 100% confident on localizing the drone)
        """ 
        for i in range(0, self.probabilities.shape[0]):
            for j in range(0, self.probabilities.shape[1]):
                if self.probabilities[i,j] > 0.999: 
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

    def visualize_nuclear_reactor(self):
        """
        generates a visualization of the nuclear reactor configuration along with probs of being at a cell. 0 -> white, 1 -> black
        """
        # set the colormap and color limits 
        plt.imshow(self.probabilities, cmap='magma', vmin=self.probabilities.min(), vmax=self.probabilities.max())

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
                    plt.text(j, i, round(self.probabilities[i,j], 3), ha="center", va="center", color="blue", fontsize=6)

        # visualizes the nuclear reactor
        plt.show()

    def visualize_nuclear_reactor_3d(self):
        """
        generates a 3D visualization of the nuclear reactor configuration with probs as the height above the surface.
        """

        # create a figure and 3D Axes
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # create a 2D array of x and y values for the 3D plot
        x, y = np.meshgrid(range(self.reactor.shape[1]), range(self.reactor.shape[0]))

        # plot the surface using the probabilities as the z values
        ax.plot_surface(x, y, self.probabilities, cmap='magma',  vmin=self.probabilities.min(), vmax=self.probabilities.max())

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
    agent = Agent(path="reactors/toyexample.txt")
    while not agent.is_terminal_state():
        print(len(agent.actions))
        agent.move_intelligently()
    print(f"The optimal action sequence is of length {len(agent.actions)} is {agent.actions}!")


    #agent = Agent() 
    # agent.move_deterministically(deactivating_path="sequences/sequence-toyexample2.txt")
    # agent = Agent()
