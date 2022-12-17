import numpy as np 

def init_nuclear_reactor_config(path = "reactors/Thor23-SA74-VERW-Schematic (Classified).txt"):
    """
    @param path (str) : the file path of the nuclear reactor configuration
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





nnps = init_nuclear_reactor_config()
print(nnps)