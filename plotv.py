import pickle as pk
import itertools
import random
import numpy as np
import copy
import networkx as nx


def compute_reward(state, num_node, demand, degree):    
    D = copy.deepcopy(state)
 
    graph = nx.from_numpy_matrix(D)
    cost = 0
    for s, d in itertools.product(range(num_node), range(num_node)):
        try:
            path_length = float(nx.shortest_path_length(graph,source=s,target=d))
        except nx.exception.NetworkXNoPath:
            path_length = float(num_node)

        cost += path_length * demand[s,d]   

    cost /= (sum(sum(demand)))   
    return cost 


class TopoSimulator(object):
    def __init__(self, n_node=8, 
                 fpath='10M_8_3.0_const3.pk3', fpath_topo='10M_8_3.0_const3_topo.pk3'):
        with open(fpath, 'rb') as f1:
            self.dataset = pk.load(f1)
        with open(fpath_topo, 'rb') as f2:
            self.toposet = pk.load(f2)

        self.max_node = n_node

    def step(self, n_node, action, no=0, demand=None, topology=None, allowed_degree=None):
        """
        :param n_node: (int) number of nodes
        :param action: (nparray) v
        :param no: (int) the index of the dataset
        :param demand: (nparray)
        :param topology: (nxdict)
        :param allowed_degree: (nparray)
        :return: cost: (float)
        """
        if demand is None:
            self.demand = self.dataset[no]['demand']
            self.allowed_degree = self.dataset[no]['allowed_degree']
            topo = self.toposet[no]
        else:
            self.demand = demand
            self.allowed_degree = allowed_degree
            topo = topology
        self.graph = nx.from_dict_of_dicts(topo)
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        Bmat = np.zeros((n_node,n_node),np.float32)
        for i in range(n_node):
            for j in range(i+1,n_node):
                deltav = np.abs(action[i]-action[j])
                Bmat[i,j] = deltav
                Bmat[j,i] = deltav
        self._graph2Pvec(Bmat) #
        obj = Bmat - np.tile(self.Pvec,(n_node,1)) - np.tile(self.Pvec.reshape(n_node,1),(1,n_node))
        adj = np.array(nx.adjacency_matrix(self.graph).todense(), np.float32)
        for i in range(n_node):
            adj[i,i] = 1
        masked_obj = (adj==0)*obj
        ind_x, ind_y = np.where(masked_obj==np.max(masked_obj))
        if len(ind_x) < 1:
            raise ValueError
        elif len(ind_x) > 1:
            s = random.randint(0, len(ind_x)-1)
            add_ind = [ind_x[s], ind_y[s]]
        else:
            add_ind = [ind_x[0], ind_y[0]]

        # Check if both nodes have available degree
        v1 = add_ind[0]
        v2 = add_ind[1]
        rm_inds = []
        if not self._check_degree(v1):
            neighbors = [n for n in self.graph.neighbors(v1)]
            h_neightbor = [Bmat[v1,n] for n in neighbors]
            v_n = neighbors[h_neightbor.index(min(h_neightbor))]
            rm_ind = [v_n,v1]
            if self._check_connectivity(rm_ind):
                self._remove_edge(rm_ind)
                rm_inds.append(rm_ind)
        if not self._check_degree(v2):
            neighbors = [n for n in self.graph.neighbors(v2)]
            h_neightbor = [Bmat[v2,n] for n in neighbors]
            v_n = neighbors[h_neightbor.index(min(h_neightbor))]
            rm_ind = [v_n,v2]
            if self._check_connectivity(rm_ind):
                self._remove_edge(rm_ind)
                rm_inds.append(rm_ind)
        if self._check_validity(add_ind):
            self._add_edge(add_ind)
        else:
            for rm_ind in rm_inds:
                self._add_edge(rm_ind)
        
        new_adj = np.array(nx.adjacency_matrix(self.graph).todense(), np.float32)
        cost = compute_reward(new_adj,self.max_node,self.demand,self.allowed_degree)
        return cost

    def step_graph(self, n_node, action, demand=None, topology=None, allowed_degree=None):
        """
        :param n_node: (int) number of nodes
        :param action: (nparray) v
        :param demand: (nparray)
        :param topology: (nxdict)
        :param allowed_degree: (nparray)
        :return: new_graph: (nxdict)
        """
        self.demand = demand
        self.allowed_degree = allowed_degree
        topo = topology
        self.graph = nx.from_dict_of_dicts(topo)
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        Bmat = np.zeros((n_node,n_node),np.float32)
        for i in range(n_node):
            for j in range(i+1,n_node):
                deltav = np.abs(action[i]-action[j])
                Bmat[i,j] = deltav
                Bmat[j,i] = deltav
        self._graph2Pvec(Bmat) #
        obj = Bmat - np.tile(self.Pvec,(n_node,1)) - np.tile(self.Pvec.reshape(n_node,1),(1,n_node))
        adj = np.array(nx.adjacency_matrix(self.graph).todense(), np.float32)
        for i in range(n_node):
            adj[i,i] = 1
        masked_obj = (adj==0)*obj
        ind_x, ind_y = np.where(masked_obj==np.max(masked_obj))
        if len(ind_x) < 1:
            raise ValueError
        elif len(ind_x) > 1:
            s = random.randint(0, len(ind_x)-1)
            add_ind = [ind_x[s], ind_y[s]]
        else:
            add_ind = [ind_x[0], ind_y[0]]

        # Check if both nodes have available degree
        v1 = add_ind[0]
        v2 = add_ind[1]
        rm_inds = []
        if not self._check_degree(v1):
            neighbors = [n for n in self.graph.neighbors(v1)]
            h_neightbor = [Bmat[v1,n] for n in neighbors]
            v_n = neighbors[h_neightbor.index(min(h_neightbor))]
            rm_ind = [v_n,v1]
            if self._check_connectivity(rm_ind):
                self._remove_edge(rm_ind)
                rm_inds.append(rm_ind)
        if not self._check_degree(v2):
            neighbors = [n for n in self.graph.neighbors(v2)]
            h_neightbor = [Bmat[v2,n] for n in neighbors]
            v_n = neighbors[h_neightbor.index(min(h_neightbor))]
            rm_ind = [v_n,v2]
            if self._check_connectivity(rm_ind):
                self._remove_edge(rm_ind)
                rm_inds.append(rm_ind)
        if self._check_validity(add_ind):
            self._add_edge(add_ind)
        else:
            for rm_ind in rm_inds:
                self._add_edge(rm_ind)
        
        return nx.to_dict_of_dicts(self.graph)


    def _add_edge(self, action):
        """
        :param action: [first_node, second_node]
        """
        self.graph.add_edge(action[0], action[1])
        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        
    def _remove_edge(self, action):
        """
        :param action: [first_node, second_node]
        """
        self.graph.remove_edge(action[0], action[1])
        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse

    def _check_validity(self, action):
        """
        Checking whether the degree constraints are still satisfied after adding the selected link.
        """
        if self.available_degree[action[0]] < 1 or self.available_degree[action[1]] < 1:
            return False
        else:
            return True

    def _check_degree(self, node):
        if self.available_degree[node] < 1:
            return False
        else:
            return True

    def _check_connectivity(self, action):
        """
        Checking whether all (src, dst) pairs are stilled connected after removing the selected link.
        """
        self.graph.remove_edge(action[0],action[1])
        """ Only considering connectivity between S-D pairs
        connected = True
        srcs, dsts = self.demand.nonzero()
        for i in range(len(srcs)):
            if not nx.has_path(self.graph,srcs[i],dsts[i]):
                connected = False
                break
        """
        if nx.is_connected(self.graph):
            connected = True
        else:
            connected = False
        self.graph.add_edge(action[0],action[1])
        return connected

    def _graph2mat(self):
        """
        Converting current graph into an adjacent matrix with edge features.
        Demands are embedded as edge features with shortest path simulation.
        """
        adj_mat = np.array(nx.adjacency_matrix(self.graph).todense())
        E_adj = adj_mat
        srcs, dsts = self.demand.nonzero()
        for i in range(len(srcs)):
            p = nx.shortest_path(self.graph,source=srcs[i],target=dsts[i])
            d = self.demand[srcs[i],dsts[i]]
            for hop in range(len(p)-1):
                E_adj[p[hop],p[hop+1]] += d
                E_adj[p[hop+1],p[hop]] += d
        return E_adj
    
    def _graph2Pvec(self, h):
        P = np.zeros((self.max_node,), np.float32)
        for v in range(self.max_node):
            if self.available_degree[v]==0:
                h_neightbor = [h[v,n] for n in self.graph.neighbors(v)]
                P[v] = min(h_neightbor)
        self.Pvec = P

def main():
    n_samples = 61
    folder = './search_8nodes_4steps/'
    n_scan = 30
    #simulator = TopoSimulator()
    alpha_range = np.linspace(start=0, stop=3, num = n_scan + 1)
    alpha_range = alpha_range.tolist()
    vscans = np.zeros((n_scan,n_scan))
    for no in range(n_samples):
        file_name = folder+'alpha_cost_'+str(no)+'.pk'
        with open(file_name,'rb') as f:
            data = pk.load(f)
        costs = []
        i = 0
        for alpha_v in alpha_range:
            #print(alpha_v)
            for alpha_i in alpha_range:
                if alpha_v == 0 or alpha_i == 0:
                    continue
                #cost = simulator.step(8,data[i]['v'],0)
                if data[i]['alpha_v'] == alpha_v and data[i]['alpha_i'] == alpha_i:
                    cost = data[i]['cost']
                else:
                    print('ERROR No.{0} i {1} alpha_v {2} alpha_i {3}'.format(
                        no, i, alpha_v, alpha_i
                    ))
                costs.append(cost)
                i += 1
        costs = np.array(costs)
        costs = costs.reshape((n_scan,n_scan))
        vscans += costs
    print(np.min(vscans))
    print(np.where(vscans == np.min(vscans)))
    vscans.tolist()
    print(vscans)
    #with open('vplot.pk', 'wb') as fp:
    #    pk.dump(costs,fp)

if __name__ == "__main__":
    main()