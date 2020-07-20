import tensorflow as tf
import gym
import networkx as nx
import copy
import itertools
import math
import random
import numpy as np
import pickle as pk
from gym import spaces

from permatch import permatch # copied from scripts/baseline folder

class TopoEnv(gym.Env):
    def __init__(self, n_nodes=8, 
                 max_action = 4, adaptive_horizon = False,
                 fpath='../../data/10000_8_4.pk3',
                 fpath_topo='../../data/10000_8_4_topo.pk3'):
        
        # Load training dataset
        with open(fpath, 'rb') as f1:
            self.dataset = pk.load(f1)
        with open(fpath_topo, 'rb') as f2:
            self.topo_dataset = pk.load(f2)

        self.n_nodes = n_nodes
        self.max_action = max_action
        # self.penalty = 0.1
        # self.rewiring_prob = 0.5

        # adaptive horizon options
        self.adaptive_horizon = adaptive_horizon
        self.episode_num = 0
        self.episode_expand = 100000
        self.max_horizon = self.n_nodes ** 2

        self.reset()

        # define action space and observation space
        self.action_space = spaces.Box(low=-1.0,high=1.0,shape=(self.n_nodes,)) #node weights
        self.observation_space = spaces.Box(
            low=0.0,high=np.float32(1e6),shape=((self.n_nodes*2+1)*self.n_nodes,)) #demand matirx & adjacent matrix & available degrees

    def reset(self,demand=None,degree=None,provide=False):
        if self.adaptive_horizon:
            self.adapt_horizon()
        self.counter = 0
        self.episode_num += 1
        self.trace_index = np.random.randint(len(self.dataset))
        if not provide:
            self.demand = self.dataset[self.trace_index]['demand']
            self.allowed_degree = self.dataset[self.trace_index]['allowed_degree']
        else:
            self.demand = demand
            self.allowed_degree = degree
        assert self.demand.shape[0] == self.demand.shape[1] # of shape [N,N]
        assert self.n_nodes == self.demand.shape[0], "Expect demand matrices of dimensions {0} but got {1}"\
            .format(self.n_nodes, self.demand.shape[0])

        self.available_degree = self.allowed_degree
        #self.edges = edge_graph.convert(demand=self.demand, allowed_degree=self.allowed_degree)
        
        self.prev_action = []
        
        self.permatch_baseline = permatch(self.n_nodes)

        # initialize graph
        self.topo_index = np.random.randint(len(self.topo_dataset))
        self.graph_dict = self.topo_dataset[self.topo_index]
        self.graph = nx.from_dict_of_dicts(self.graph_dict)
        
        self.last_graph = copy.deepcopy(self.graph)
        
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        
        adj = np.array(nx.adjacency_matrix(self.graph).todense(), np.float32)
        expand_availdeg = self.available_degree[np.newaxis,:]
        demand_norm = self.demand/(np.max(self.demand)+1e-7)
        obs = np.concatenate((demand_norm,adj,expand_availdeg),axis=0)
        obs = obs.flatten()
        return obs

    def step(self, action):
        """
        :param action: [weights for nodes..., stop]
        :return: next_state, reward, done, {}
        """
        assert self.action_space.contains(action), "action type {} is invalid.".format(type(action))

        expand_action = np.tile(action, (self.n_nodes, 1))
        obj = np.abs(expand_action - expand_action.T)
        #self._graph2Pvec(Bmat)
        #obj = Bmat - np.tile(self.Pvec,(self.n_nodes,1)) - np.tile(self.Pvec.reshape(self.n_nodes,1),(1,self.n_nodes))
        adj = np.array(nx.adjacency_matrix(self.graph).todense(), np.float32)
        mask = adj + np.identity(self.n_nodes, np.float32)
        masked_obj = (mask == 0) * obj
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

        stop = self.counter >= (self.max_action - 1)

        rm_inds = []
        if not self._check_degree(v1):
            neighbors = [n for n in self.graph.neighbors(v1)]
            h_neightbor = [obj[v1,n] for n in neighbors]
            v_n = neighbors[h_neightbor.index(min(h_neightbor))]
            rm_ind = [v_n,v1]
            if self._check_connectivity(rm_ind):
                self._remove_edge(rm_ind)
                #print("remove edge ({0},{1})".format(v_n,v1))
                rm_inds.append(rm_ind)
        if not self._check_degree(v2):
            neighbors = [n for n in self.graph.neighbors(v2)]
            h_neightbor = [obj[v2,n] for n in neighbors]
            v_n = neighbors[h_neightbor.index(min(h_neightbor))]
            rm_ind = [v_n,v2]
            if self._check_connectivity(rm_ind):
                self._remove_edge(rm_ind)
                #print("remove edge ({0},{1})".format(v_n,v2))
                rm_inds.append(rm_ind)
        if self._check_validity(add_ind):
            self._add_edge(add_ind)
            #print("add edge ({0},{1})".format(v1,v2))
        else:
            for rm_ind in rm_inds:
                self._add_edge(rm_ind)

        reward = 0.0

        adj = np.array(nx.adjacency_matrix(self.graph).todense(), np.float32)
        expand_availdeg = self.available_degree[np.newaxis,:]
        demand_norm = self.demand/(np.max(self.demand)+1e-7) * 4
        obs = np.concatenate((demand_norm,adj,expand_availdeg),axis=0)
        obs = obs.flatten()
        self.counter += 1
        if stop:
            reward = 100 * self._cal_reward_against_permatch()
            print("[Horizon {0}] [Episode {1}] [Reward {2}]".format(
                self.max_action, self.episode_num,reward))
            self.reset()
        
        return obs, reward, stop, {}
    
    def _cal_step_reward(self):
        last_score = 0
        cur_score = 0
        for s, d in itertools.product(range(self.n_nodes), range(self.n_nodes)):
            try:
                last_path_length = float(nx.shortest_path_length(self.last_graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                last_path_length = float(self.n_nodes)

            try:
                cur_path_length = float(nx.shortest_path_length(self.graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                cur_path_length = float(self.n_nodes)

            last_score += last_path_length * self.demand[s][d]
            cur_score += cur_path_length * self.demand[s][d]
        
        last_score /= (sum(sum(self.demand)))
        cur_score /= (sum(sum(self.demand)))
        #last_score /= (sum(sum(self.demand)) * math.sqrt(self.n_nodes))
        #cur_score /= (sum(sum(self.demand)) * math.sqrt(self.n_nodes))
        return last_score - cur_score

    def _cal_reward_against_permatch(self):
        nn_score = 0
        permatch_score = 0
        #last_adj = np.array(nx.adjacency_matrix(self.last_graph).todense())
        #permatch_new_adj = self.permatch_baseline.n_steps_matching(self.demand,last_adj,self.available_degree)
        permatch_new_graph = self.permatch_baseline.n_steps_matching(
            self.demand,self.last_graph,self.allowed_degree,self.max_action) # weighted matching

        for s, d in itertools.product(range(self.n_nodes), range(self.n_nodes)):
            try:
                permatch_path_length = float(nx.shortest_path_length(permatch_new_graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                permatch_path_length = float(self.n_nodes)

            try:
                nn_path_length = float(nx.shortest_path_length(self.graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                nn_path_length = float(self.n_nodes)

            permatch_score += permatch_path_length * self.demand[s][d]
            nn_score += nn_path_length * self.demand[s][d]
        
        #permatch_score /= (sum(sum(self.demand)))
        #nn_score /= (sum(sum(self.demand)))
        #last_score /= (sum(sum(self.demand)) * math.sqrt(self.n_nodes))
        #cur_score /= (sum(sum(self.demand)) * math.sqrt(self.n_nodes))
        return (permatch_score - nn_score)/permatch_score


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
        P = np.zeros((self.n_nodes,), np.float32)
        for v in range(self.n_nodes):
            if self.available_degree[v]==0:
                h_neightbor = [h[v,n] for n in self.graph.neighbors(v)]
                P[v] = min(h_neightbor)
        self.Pvec = P

    def _demand_matrix_extend(self):
        """Converting demand matrix into N x N x N x N sparse matrix
        """
        sparse_demand = np.zeros([self.n_nodes**2,self.n_nodes,self.n_nodes],np.float32)
        srcs, dsts = self.demand.nonzero()
        for i in range(len(srcs)):
            s = srcs[i]
            d = dsts[i]
            demand = self.demand[s,d]
            sparse_demand[s*self.n_nodes+d,d,:] -= demand
            sparse_demand[s*self.n_nodes+d,:,s] += demand
        return sparse_demand

    def _adj_extend(self, adj):
        """
        Encoding available degree into adjacency matrix
        """
        deg_diag = np.diag(self.available_degree)
        return adj + deg_diag

    def adapt_horizon(self):
        if ((self.episode_num != 0)
            and (self.episode_num % self.episode_expand == 0)
            and (self.max_action < self.max_horizon)):

            self.max_action *= 2
