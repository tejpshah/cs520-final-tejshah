import numpy as np 
import matplotlib.pyplot as plt 

class Agent():
    def __init__(self, path ="reactors/Thor23-SA74-VERW-Schematic (Classified).txt"):

        # this stores the path of the nuclear reactor 
        self.path = path

        # this represents the nuclear reactor as a 2D matrix
        self.reactor = self.init_nuclear_reactor_config()

        # this represents probability of being at a cell as a matrix
        self.probabilities = self.init_probability_matrix()

        # this will generate a visualization of the initial state of the board
        self.visualize_nuclear_reactor()

    def init_nuclear_reactor_config(self):
        """
        generates a 2D numpy matrix that represents the relevant configuration of a nuclear reactor. 

        @param path : the file path of the nuclear reactor configuration
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

        @param reactor : represents the configuration of the nuclear reactor {0->unblocked cell, 1 ->blocked cell}
        @returns transition_matrix : represents the probability of drone being at cell (i,j). 
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

if __name__ == "__main__":
    agent = Agent(path="reactors/toyexample.txt") 
