import numpy as np
import networkx as nx

from polyfit.hiertopo import HierTopoPolyn
from baseline.dijkstra_greedy import DijGreedyAlg

class SafeHierTopoAlg(object):
    def __init__(self, n_node, n_degree, n_iter, n_maxstep, k):
        self.n_node = n_node
        self.n_degree = n_degree
        
        self.hiertopo_model = HierTopoPolyn(n_node, n_degree, n_iter, n_maxstep, k)
        self.rgreedy_model = DijGreedyAlg(n_node, n_degree)
        
        self.cntr = 0
        self.period = 5
    
    def single_move(self, demand, graph, cand, alpha):
        is_end_ht, e_ht, cand_ht = self.hiertopo_model.single_move_wo_replace(demand, graph, cand, alpha)
        is_end_rg, e_rg, cand_rg = self.rgreedy_model.single_move_wo_replace(demand, graph, cand)
        
        self.fallback(is_end_ht, e_ht, cand_ht, is_end_rg, e_rg, cand_rg)
        
    def fallback(self, is_end_ht, e_ht, cand_ht, is_end_rg, e_rg, cand_rg):
        return self.fallback_period(is_end_ht, e_ht, cand_ht, is_end_rg, e_rg, cand_rg)
        
    def fallback_period(self, is_end_ht, e_ht, cand_ht, is_end_rg, e_rg, cand_rg):
        if is_end_ht and is_end_rg:
            return True, 0, cand_ht, cand_rg
        if is_end_ht:
            return True, 0, cand_ht, cand_rg
        if is_end_rg:
            return False, e_ht, cand_ht, cand_rg
        
        # both algorithm has normal output
        if self.cntr % self.period == 0:
            # use Hiertopo's decision
            if e_ht in cand_rg:
                e_idx = cand_rg.index(e_ht)
                del cand_rg[e_idx]
            return True, e_ht, cand_ht, cand_rg
        else:
            # use routing-greedy decision:
            if e_rg in cand_ht:
                e_idx = cand_ht.index(e_rg)
                del cand_ht[e_idx]
            return True, e_rg, cand_ht, cand_rg
        
    def run(self, demand):
        G = nx.Graph()
        G.add_nodes_from(list(range(self.n_node)))
        adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
        #adj = permatch_model.matching(demand, np.ones((n_node,)) * (n_degree-1))
        #graph = nx.from_numpy_matrix(adj)
        degree = np.sum(adj, axis=-1)
        
        