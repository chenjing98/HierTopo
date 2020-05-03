import networkx as nx
import numpy as np
import itertools
import copy

class optimal(object):
    def __init__(self):
        pass

    def compute_optimal(self,num_node,graph,demand,allowed_degree,steps=1):
        """
        Args:
            num_node: the total number of nodes in the network
            graph: networkx object, current topology
            demand: traffic demands in matrix form
            allowed_degree: a vector for degree constraints
        Returns:
            optimal actions: the edge between a pair of nodes to add
            optimal V: in order to pretrain the RL model supervisedly
        """
        self.num_node = num_node
        self.graph = copy.deepcopy(graph)
        self.demand = demand
        self.allowed_degree = allowed_degree
        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        self.origin_cost = self.cal_cost()

        self.inf = 1e6
        
        costs = []
        neighbors = []
        for i in range(num_node-1):
            for j in range(i+1,num_node):
                if graph.has_edge(i,j):
                    # already has this edge
                    costs.append(self.inf)
                    neighbors.append(dict())
                else:
                    # consider how good it is to add edge (i,j)
                    cost, neigh = self.consideration(i,j)
                    costs.append(cost)
                    neighbors.append(neigh)
        if min(costs) < self.origin_cost:
            ind = costs.index(min(costs))
            neigh = neighbors[ind]
            bestaction = edge_to_node(num_node,ind)
            v1 = bestaction[0]
            v2 = bestaction[1]
            self._add_edge(v1,v2)
            if "left" in neigh:
                self._remove_edge(neigh["left"],v1)
            if "right" in neigh:
                self._remove_edge(neigh["right"],v2)

            return bestaction, neigh, self.graph
        else:
            return [], dict(), graph

    def consturct_v(self, best_action, neighbor_to_remove):
        """Constructiong a vector V for supervised learning.
        Args: 
            best_action: (list) the node pair to be added
            neighbor_to_remove: (dict) keys may contains "left" and "right"
        
        Returns:
            v: (nparray) a vector V for voltages
        """
        v = np.zeros((self.num_node,),np.float32)
        if len(best_action) == 0:
            return v
        upper_demand = np.triu(self.demand,k=1)
        lower_demand = np.tril(self.demand,k=-1)        
        sum_demand_u = np.sum(upper_demand)
        sum_demand_l = np.sum(lower_demand)
        v1 = best_action[0]
        v2 = best_action[1]
        v[v1] = -1.0
        v[v2] = 1.0
        if "left" in neighbor_to_remove:
            v[neighbor_to_remove["left"]] = -0.5
        if "right" in neighbor_to_remove:
            v[neighbor_to_remove["right"]] = 0.5
        if sum_demand_l > sum_demand_u:
            v = -v
        return v

    def consideration(self,v1,v2):
        neighbors1 = [n for n in self.graph.neighbors(v1)]
        neighbors2 = [n for n in self.graph.neighbors(v2)]
        self._add_edge(v1,v2)
        costs = []
        if not self._check_degree(v1) and not self._check_degree(v2): 
            for left in range(len(neighbors1)):
                for right in range(len(neighbors2)):
                    n_l = neighbors1[left]
                    n_r = neighbors2[right]
                    self._remove_edge(n_l,v1)
                    self._remove_edge(n_r,v2)
                    if not nx.is_connected(self.graph):
                        costs.append(self.inf)
                    else:
                        cost = self.cal_cost()
                        costs.append(cost)
                    self._add_edge(n_l,v1)
                    self._add_edge(n_r,v2)
            ind = costs.index(min(costs))
            ind_left = ind // len(neighbors2)
            ind_right = ind % len(neighbors2)
            neigh = {"left":neighbors1[ind_left],"right":neighbors2[ind_right]}
        elif not self._check_degree(v1):
            for left in range(len(neighbors1)):
                n_l = neighbors1[left]
                self._remove_edge(n_l,v1)
                if not nx.is_connected(self.graph):
                    costs.append(self.inf)
                else:
                    cost = self.cal_cost()
                    costs.append(cost)
                self._add_edge(n_l,v1)
            ind_left = costs.index(min(costs))
            neigh = {"left":neighbors1[ind_left]}
        elif not self._check_degree(v2):
            for right in range(len(neighbors2)):
                n_r = neighbors2[right]
                self._remove_edge(n_r,v2)
                if not nx.is_connected(self.graph):
                    costs.append(self.inf)
                else:
                    cost = self.cal_cost()
                    costs.append(cost)
                self._add_edge(n_r,v2)
            ind_right = costs.index(min(costs))
            neigh = {"right":neighbors2[ind_right]}
        else:
            cost = self.cal_cost()
            costs.append(cost)
            neigh = dict()
        min_cost = min(costs)
        self._remove_edge(v1,v2)
        return min_cost, neigh
    
    def cal_cost(self):
        cost = 0
        for s, d in itertools.product(range(self.num_node), range(self.num_node)):
            try:
                path_length = float(nx.shortest_path_length(self.graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                path_length = float(self.num_node)

            cost += path_length * self.demand[s][d]
        
        return cost

    def _add_edge(self, v1, v2):
        self.graph.add_edge(v1,v2)

        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse

    def _remove_edge(self, v1, v2):
        self.graph.remove_edge(v1, v2)
        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse

    def _check_degree(self, node):
        if self.available_degree[node] < 0:
            return False
        else:
            return True

def edge_to_node(node_num,e):
    c = 0
    for i in range(node_num-1):
        for j in range(i+1,node_num):
            if c == e:
            #if ((i*(2*node_num-1-i)/2-1+j-i)== e):
                return [i,j]
            c += 1