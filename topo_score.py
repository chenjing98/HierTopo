import networkx
import numpy as np

def topo_score(demand, topo):
    path_length = np.array(networkx.shortest_path_length(topo))
    return np.multiply(demand, path_length)
    