
from polyfit.hiertopo import HierTopoPolyn
from baseline.dijkstra_greedy import DijGreedyAlg

class SafeHierTopoAlg(object):
    def __init__(self, n_node, n_degree, n_iter, n_maxstep, k):
        self.n_node = n_node
        self.n_degree = n_degree
        
        self.hiertopo_model = HierTopoPolyn(n_node, n_degree, n_iter, n_maxstep, k)
        self.rgreedy_model = DijGreedyAlg(n_node, n_degree)
    
    def single_move(self):
        pass