import networkx as nx 
import numpy as np
import matplotlib.pyplot as plt

import edge_graph


# edges = edge_graph.convert(np.zeros((5,5)), np.zeros(5))
# print(nx.adjacency_matrix(edges).todense())
# nx.draw_circular(edges)
# plt.show()

a = np.array([[0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
 [1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
 [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
 [1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
 [1, 1, 0, 0, 0, 1, 1, 1, 1, 0],
 [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
 [1, 0, 0, 1, 1, 1, 0, 0, 1, 1],
 [0, 1, 1, 0, 1, 1, 0, 0, 1, 1],
 [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
 [0, 0, 1, 1, 0, 1, 1, 1, 1, 0]])

b = np.array([[0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
 [1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
 [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
 [0, 1, 0, 1, 0, 1, 1, 0, 1, 1],
 [1, 0, 1, 0, 1, 1, 0, 1, 1, 0],
 [0, 1, 0, 1, 1, 0, 1, 1, 0, 1],
 [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
 [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
 [1, 1, 0, 1, 1, 0, 1, 0, 1, 0],
 [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]])

print(np.linalg.eig(a))
print(np.linalg.eig(b))