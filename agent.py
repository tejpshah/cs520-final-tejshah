import random 
import heapq
import numpy as np 
import matplotlib.pyplot as plt 

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

        # this runs the debug command
        # self.debug()

    # "INTELLIGENT" LOGIC TO MOVE THE AGENT WITH RESPECT TO THE CORRECT SEQUENCE
    
    def a_star(self):

        print("\n ------ INITIALIZE THE A STAR ALGORITHM -----\n")

        def cost(curr_probs, next_probs):
            """ @returns the cost up to the point as number of steps taken"""
            return 1 - (next_probs.max() - curr_probs.max())

        def heuristic(probabilities):
            """ @returns negative log likelihood of the cell with the highest probability"""
            return -np.log(probabilities.max())
        
        print(f"\nSTARTING THE A STAR ALGORITHM...")

        # initialize the heap list and visited set
        heap, visited = list(), set() 

        # initialize heap with ( cost(s), ( s, seq(s) ) )
        s0 = AStarTuple(0 + heuristic(self.probabilities), self.probabilities)
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

                print(f"\nIf we move with action {action}, we get the new proabilities:")
                print(next_probs)

                # if we've already visited this state before continue 
                if tuple(next_probs.flatten()) not in visited: 

                    total_cost, next_seq= heuristic(next_probs) + cost(curr_state.probabilities, next_probs), curr_seq + [action]

                    s1 = AStarTuple(total_cost, next_probs)
                    heapq.heappush(heap, (s1, next_seq))

                    print(f"\nWe are pushing this information to the heap...")
                    print(f"The total cost: {total_cost}")
                    print(f"The next sequence: {next_seq}")
                    print(f"The new probabilities:")
                    print(s1.probabilities)

        return heap 

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
    agent = Agent(path="reactors/toyexample3.txt")
    #agent = Agent()
    agent.a_star()
    print(f"The optimal action sequence is of length {len(agent.actions)} is {agent.actions}!")

    #agent = Agent() 
    # agent.move_deterministically(deactivating_path="sequences/sequence-toyexample2.txt")
    # agent = Agent()
