import numpy as np 
import agent as Agent 

def generate_mazes():
  mazes = []
  for i in range(2):
    for j in range(2):
      for k in range(2):
        for l in range(2):
          for m in range(2):
            for n in range(2):
              for o in range(2):
                for p in range(2):
                  for q in range(2):
                    maze = np.array([[i, j, k], [l, m, n], [o, p, q]])
                    mazes.append(maze)
  return mazes

mazes = generate_mazes() 
for maze in mazes:
    agent = Agent("")
