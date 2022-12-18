import numpy as np 
import matplotlib.pyplot as plt 

def init_nuclear_reactor_config(path = "reactors/Thor23-SA74-VERW-Schematic (Classified).txt"):
    """
    generates a 2D numpy matrix that represents the relevant configuration of a nuclear reactor. 

    @param path : the file path of the nuclear reactor configuration
    @returns reactor : 2D numpy matrix with the configuration of the reactor {"_" -> 0, "X" -> 1}
    """

    # reads in the file as a list of strings
    with open(path, "r") as f:
        lines = f.readlines() 
    
    # remove leading/trailing white spaces in the text file
    lines = [list(line.strip()) for line in lines]

    # {"_" -> 0, "X" -> 1} conversion for the numpy matrix 
    lines = [[1 if x == "X" else 0 for x in line] for line in lines]

    # convert the list of lists to a 2D numpy array 
    reactor = np.array(lines)

    # return a 2D numpy array of the nuclear reactor configuration
    return reactor 

def init_probability_matrix(reactor):
    """
    generates initial transition matrix for where the drone can be as 1 / [# of white cells (aka reactor with 0)] in each location of matrix

    @param reactor : represents the configuration of the nuclear reactor {0->unblocked cell, 1 ->blocked cell}
    @returns transition_matrix : represents the probability of drone being at cell (i,j). 
    """

    # count the number of white cells in the nuclear reactor
    num_white_cells = (reactor == 0).sum()

    print(f"The number of white cells are {num_white_cells}")

    # initial probability matrix of same shape as reactor with all being equally likely
    transition_matrix = np.ones(reactor.shape) * (1 / num_white_cells)

    # set the probability of black cells in reactor to 0
    transition_matrix[reactor == 1] = 0 

    # return the probability matrix
    return transition_matrix 

def visualize_nuclear_reactor(reactor, probabilities):
    """
    generates a visualization of the nuclear reactor configuration along with probs of being at a cell. 0 -> white, 1 -> black
    
    @param reactor : represents the configuration of the nuclear reactor
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
    for i in range(reactor.shape[0]):
        for j in range(reactor.shape[1]):
            if reactor[i,j] != 1:
                plt.text(j, i, round(probabilities[i,j], 3), ha="center", va="center", color="blue", fontsize=6)

    # visualizes the nuclear reactor
    plt.show()

def move_down(reactor, probabilities):
    


if __name__ == "__main__":

    # initialize nuclear reactor and transition matrix
    reactor = init_nuclear_reactor_config(path="reactors/toy.txt")
    probabilities = init_probability_matrix(reactor)

    # generate heatmap of probability transition matrix
    visualize_nuclear_reactor(reactor, probabilities)