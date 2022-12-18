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
        self.debug()

# INITIALIZATION FUNCTIONS FOR NUCLEAR REACTOR, PROBABILITIES, & INVALID ACTIOSN

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



# DEBUGGING FUNCTIONS FOR PRINTING OUTPUT TO TERMINAL AND VISUALIZING GAME STATE

    def visualize_nuclear_reactor(self):
        """
        generates a visualization of the nuclear reactor configuration along with probs of being at a cell. 0 -> white, 1 -> black
        @param reactor : represents the configuration of the nuclear reactor
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

    def debug(self):
        """
        prints out to terminal the probabilities, nuclear reactor, and other debugging information.
        """
        
        print(f"\nTHE REACTOR IS:")
        print(self.reactor)

        print(f"\nTHE PROBABITIES ARE:")
        print(self.probabilities)
        
        self.visualize_nuclear_reactor()

if __name__ == "__main__":
    agent = Agent() 
