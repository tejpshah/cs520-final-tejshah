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

def visualize_nuclear_reactor(reactor):
    """
    generates a visualization of the nuclear reactor configuration. 0 -> white, 1 -> black

    @param reactor : represents the configuration of the nuclear reactor
    """

    # set the colormap and color limits 
    cmap = plt.get_cmap("binary")
    plt.imshow(reactor, cmap=cmap)

    # removes the tick marks and labels
    plt.gca().tick_params(which="both", length=0)
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    # sets the tick labels at center of each cell to be the values in each cell
    for i in range(reactor.shape[0]):
        for j in range(reactor.shape[1]):
            plt.text(j, i, reactor[i,j], ha="center", va="center", color="white" if reactor[i,j] == 0 else "black")

    # visualizes the nuclear reactor
    plt.show()

nnps = init_nuclear_reactor_config()
visualize_nuclear_reactor(nnps)