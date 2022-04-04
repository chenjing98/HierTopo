
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
    
    def single_move(self, demand, graph, degree, cand, alpha):
        is_end_ht, n0_ht, n1_ht, cand_ht = self.hiertopo_model.single_move_wo_replace(demand, graph, degree, cand, alpha)
        is_end_rg, n0_rg, n1_rg, cand_rg = self.rgreedy_model.single_move_wo_replace(demand, graph, degree, cand)
        
        self.fallback(is_end_ht, n0_ht, n1_ht, cand_ht, is_end_rg, n0_rg, n1_rg, cand_rg)
        
    def fallback(self, is_end_ht, n0_ht, n1_ht, cand_ht, is_end_rg, n0_rg, n1_rg, cand_rg):
        return self.fallback_period(is_end_ht, n0_ht, n1_ht, cand_ht, is_end_rg, n0_rg, n1_rg, cand_rg)
        
    def fallback_period(self, is_end_ht, n0_ht, n1_ht, cand_ht, is_end_rg, n0_rg, n1_rg, cand_rg):
        if is_end_ht and is_end_rg:
            return True, 0, 0, cand_ht, cand_rg
        if is_end_ht:
            return True, 0, 0, cand_ht, cand_rg
        if is_end_rg:
            return False, n0_ht, n1_ht, cand_ht, cand_rg
        
        # both algorithm has normal output
        if self.cntr % self.period == 0:
            # use Hiertopo's decision
            
            return True, n0_ht, n1_ht