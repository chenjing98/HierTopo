from turtle import update
import numpy as np
import copy
import itertools
import networkx as nx

class DijGreedyAlg (object):
    def __init__(self, n_nodes, max_degree) -> None:
        self.n_nodes = n_nodes
        self.max_degree = max_degree
        self.inf = max(100, n_nodes)
        
    def reset(self):
        pass

    def topo_scratch(self, demand, degree):
        allowed_degree = copy.deepcopy(degree)
        demand_vec = []
        for i in range(self.n_nodes - 1):
            for j in range(i + 1, self.n_nodes):
                demand_vec.append(demand[i,j] + demand[j,i])
                
        state = np.zeros((self.n_nodes, self.n_nodes))
        graph = nx.Graph()
        graph.add_nodes_from(list(range(self.n_nodes)))
        plen_vec = self.update_plen(graph)
        crit_vec = []
        for i in range(int(self.n_nodes * (self.n_nodes - 1) / 2)):
            crit_vec.append(demand_vec[i] * plen_vec[i])
        
        while(True):
            # print("max crit vec: {}".format(max(crit_vec)))
            if max(crit_vec) <= 0:
                break
            # choose edge
            e = crit_vec.index(max(crit_vec))
            n = self.edge_to_node(e)
            n1 = n[0]
            n2 = n[1]
            
            #make this edge not be used again
            demand_vec[e] = - self.inf
            
            if allowed_degree[n1] > 0 and allowed_degree[n2] > 0:
                state[n1,n2] = 1
                state[n2,n1] = 1
                allowed_degree[n1] -= 1
                allowed_degree[n2] -= 1
                graph.add_edge(n1, n2)
                plen_vec = self.update_plen(graph)
            crit_vec = []
            for i in range(int(self.n_nodes * (self.n_nodes - 1) / 2)):
                crit_vec.append(demand_vec[i] * plen_vec[i])
            
        return state
        
        
    
    def topo_nsteps(self, demand, graph, degree, n_steps):
        allowed_degree = copy.deepcopy(degree)
        demand_vec = []
        for i in range(self.n_nodes - 1):
            for j in range(i + 1, self.n_nodes):
                demand_vec.append(demand[i,j] + demand[j,i])
        plen_vec = self.update_plen(graph)
        crit_vec = []
        for i in range(int(self.n_nodes * (self.n_nodes - 1) / 2)):
            crit_vec.append(demand_vec[i] * plen_vec[i])
        new_graph = copy.deepcopy(graph)
        
        step = 0
        while step < n_steps:
            if max(crit_vec) < 0:
                break
            # choose edge
            e = crit_vec.index(max(crit_vec))
            n = self.edge_to_node(e)
            n1 = n[0]
            n2 = n[1]
            
            #make this edge not be used again
            demand_vec[e] = - self.inf
            
            if allowed_degree[n1] > 0 and allowed_degree[n2] > 0:
                step += 1
                allowed_degree[n1] -= 1
                allowed_degree[n2] -= 1
                new_graph.add_edge(n1, n2)
                
                plen_vec = self.update_plen(new_graph)
                crit_vec = []
                for i in range(int(self.n_nodes * (self.n_nodes - 1) / 2)):
                    crit_vec.append(demand_vec[i] * plen_vec[i])
                    
        return new_graph
    
    
    def update_plen(self, graph):
        plen_vec = []
        for i in range(self.n_nodes - 1):
            for j in range(i + 1, self.n_nodes):
                try:
                    path_length = float(nx.shortest_path_length(graph,source=i,target=j))
                except nx.exception.NetworkXNoPath:
                    path_length = float(self.inf)
                
                plen_vec.append(path_length - 1)
        return plen_vec
            
    #the order of edge ---> the order of two nodes
    def edge_to_node(self,e):
        for i in range(self.n_nodes - 1):
            for j in range(i + 1, self.n_nodes):
                if ((i* (2 * self.n_nodes - 1 - i) / 2 - 1 + j - i)== e):
                    return [i,j]