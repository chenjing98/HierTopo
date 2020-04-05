import networkx as nx
import copy
import itertools
import math
import random
import numpy as np
import pickle as pk

class TopoEnv(object):

    def __init__(self, n_node=8, max_action=50, fpath='10M_8_3.0_const3.pk3'):
        with open(fpath, 'rb') as f:
            self.dataset = pk.load(f)

        #self.connect_penalty = 0.1
        #self.degree_penalty = 0.1
        self.penalty = 0.1
        self.max_action = max_action
        self.max_node = n_node

        # isolated random number generator
        self.np_random = np.random.RandomState()

        self.reset()

        """
        # define action space and observation space
        self.action_space = spaces.Box(low=0.0,high=1.0,shape=(self.max_node+1,)) #node weights & stop sign
        self.observation_space = spaces.Box(
            low=0.0,high=np.float32(1e6),shape=(self.max_node,self.max_node+1)) #featured adjacent matrix & available degrees
        """

    def reset(self,demand=None,degree=None,provide=False):
        self.counter = 0
        self.trace_index = np.random.randint(len(self.dataset))
        if not provide:
            self.demand = self.dataset[self.trace_index]['demand']
            self.allowed_degree = self.dataset[self.trace_index]['allowed_degree']
        else:
            self.demand = demand
            self.degree = degree
        assert self.demand.shape[0] == self.demand.shape[1] # of shape [N,N]
        assert self.max_node == self.demand.shape[0], "Expect demand matrices of dimensions {0} but got {1}"\
            .format(self.max_node, self.demand.shape[0])

        self.available_degree = self.allowed_degree
        #self.edges = edge_graph.convert(demand=self.demand, allowed_degree=self.allowed_degree)
        
        # Initialize a path graph
        self.graph = nx.path_graph(self.max_node)

        #E_adj = self._graph2mat()
        #obs = np.concatenate((E_adj,self.available_degree[:,np.newaxis]),axis=-1)
        obs = (self.graph, self.demand)
        return obs

    def step(self, action):
        """
        :param action: [weights for nodes..., stop]
        :return: next_state, reward, done, {}
        """
        # assert self.action_space.contains(action), "action type {} is invalid.".format(type(action))
        self.last_graph = copy.deepcopy(self.graph)

        # Start a new episode
        #stop = action[-1] or self.counter >= self.max_action
        stop = self.counter >= self.max_action

        #node_weights = action[:-1]
        Bmat = action
        self._graph2Pvec(Bmat)
        obj = Bmat - np.tile(self.Pvec,(self.max_node,1)) - np.tile(self.Pvec.reshape(self.max_node,1),(1,self.max_node))
        adj = np.array(nx.adjacency_matrix(self.graph).todense())
        for i in range(self.max_node):
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
        #n_max = node_weights.index(max(node_weights))
        #n_min = node_weights.index(min(node_weights))
        #node_weights = np.array(node_weights)
        #add_ind = [n_max,n_min]

        # Check if both nodes have available degree
        v1 = add_ind[0]
        v2 = add_ind[1]
        rm_inds = []
        cost = 0
        if not self._check_degree(v1):
            neighbors = [n for n in self.graph.neighbors(v1)]
            h_neightbor = [Bmat[v1,n] for n in neighbors]
            v_n = neighbors[h_neightbor.index(min(h_neightbor))]
            rm_ind = [v_n,v1]
            cost += min(h_neightbor)
            rm_inds.append(rm_ind)
            if self._check_connectivity(rm_ind):
                self._remove_edge(rm_ind)
                print("remove edge ({0},{1}".format(v_n,v1))
        if not self._check_degree(v2):
            neighbors = [n for n in self.graph.neighbors(v2)]
            h_neightbor = [Bmat[v2,n] for n in neighbors]
            v_n = neighbors[h_neightbor.index(min(h_neightbor))]
            rm_ind = [v_n,v2]
            cost += min(h_neightbor)
            rm_inds.append(rm_ind)
            if self._check_connectivity(rm_ind):
                self._remove_edge(rm_ind)
                print("remove edge ({0},{1})".format(v_n,v2))
        if self._check_validity(add_ind):
            if Bmat[v1,v2] > cost:
                self._add_edge(add_ind)
                print("add edge ({0},{1})".format(v1,v2))
            else:
                for rm_ind in rm_inds:
                    self._add_edge(rm_ind)
                print("totalstep {}".format(self.counter))
                stop = True

        self.counter += 1
        #print("counter:{}".format(self.counter))
        obs = (self.graph, self.demand)
        if stop:
            self.reset()

        return obs, stop#, reward, stop, {}

    def seed(self, seed):
        self.np_random.seed(seed)
    
    def _cal_step_reward(self):
        last_score = 0
        cur_score = 0
        for s, d in itertools.product(range(self.max_node), range(self.max_node)):
            try:
                last_path_length = float(nx.shortest_path_length(self.last_graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                last_path_length = float(self.max_node)

            try:
                cur_path_length = float(nx.shortest_path_length(self.graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                cur_path_length = float(self.max_node)

            last_score += last_path_length * self.demand[s][d]
            cur_score += cur_path_length * self.demand[s][d]
        
        last_score /= (sum(sum(self.demand)))
        cur_score /= (sum(sum(self.demand)))
        #last_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        #cur_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        return last_score - cur_score


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


class TopoOperator(object):
    def __init__(self, n_node, infinity=1e6):
        self.max_node = n_node
        self.inf = infinity

    def reset(self, n_node=None):
        if n_node is not None:
            self.max_node = n_node

    def h_predict(self, graph, demand):
        self.demand = demand
        self.graph = graph
        h = np.zeros((self.max_node,self.max_node),np.float32)
        srcs, dsts = self.demand.nonzero()
        for i in range(len(srcs)):
            h += self.demand[srcs[i],dsts[i]]*self._h_per_sd(srcs[i], dsts[i])
        return h

    def _h_per_sd(self, source, destination):
        h_sd = np.zeros((self.max_node,self.max_node),np.float32)
        for v1 in range(self.max_node):
            for v2 in range(self.max_node):
                h_sd[v1,v2] = self._h_per_sd_per_edge(source, destination, v1, v2)
        return h_sd

    def _h_per_sd_per_edge(self, source, destination, v1, v2):
        if self.graph.has_edge(v1,v2):
            le = nx.shortest_path_length(self.graph,source,destination)
            self.graph.remove_edge(v1,v2)
            if nx.has_path(self.graph,source,destination):
                l = nx.shortest_path_length(self.graph,source,destination)
            else:
                l = self.inf
            self.graph.add_edge(v1,v2)
        else:
            l = nx.shortest_path_length(self.graph,source,destination)
            self.graph.add_edge(v1, v2)
            le = nx.shortest_path_length(self.graph,source,destination)
            self.graph.remove_edge(v1,v2)
        return l-le