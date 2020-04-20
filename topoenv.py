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

from baseline_new.permatch import permatch

class TopoEnv(gym.Env):

    def __init__(self, n_node=8, fpath='10M_8_3.0_const3.pk3'):
        with open(fpath, 'rb') as f:
            self.dataset = pk.load(f)

        self.connect_penalty = 0.1
        self.degree_penalty = 0.1
        self.penalty = 0.1
        self.max_action = 1
        self.max_node = n_node
        self.init_degree = 2

        self.episode_num = 0
        self.episode_expand = 2000000
        self.max_horizon = self.max_node ** 2
        # isolated random number generator
        self.np_random = np.random.RandomState()

        self.reset()

        # define action space and observation space
        self.action_space = spaces.Box(low=0.0,high=1.0,shape=(self.max_node**2,)) #node weights
        self.observation_space = spaces.Box(
            low=0.0,high=np.float32(1e6),shape=(self.max_node**2+1,self.max_node)) #demand matirx & adjacent matrix & available degrees

    def reset(self,demand=None,degree=None,provide=False):
        self.adaptive_horizon()
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
        assert self.max_node == self.demand.shape[0], "Expect demand matrices of dimensions {0} but got {1}"\
            .format(self.max_node, self.demand.shape[0])

        self.available_degree = self.allowed_degree
        #self.edges = edge_graph.convert(demand=self.demand, allowed_degree=self.allowed_degree)
        
        self.prev_action = []
        
        self.permatch_baseline = permatch(self.max_node)
        
        # Initialize the graph with a stochastic connected graph
        try:
            self.graph = nx.connected_watts_strogatz_graph(
                self.max_node,self.init_degree,tries=50,seed=self.np_random.randint(10))
            print("======= initial: watts strogatz graph =======")
        except nx.NetworkXError:
            # initialization with path graph
            self.graph = nx.path_graph(self.max_node)
            print("============ initial: path graph =============")
        adj = np.array(nx.adjacency_matrix(self.graph).todense(), np.float32)
        """
        sp_demand = self._demand_matrix_extend()
        #expand_adj = np.tile(adj[np.newaxis, np.newaxis,:,:], (self.max_node,1,1,1))
        #enc_adj = self._adj_extend(adj)
        expand_adj = adj[np.newaxis,:,:]
        obs = np.concatenate((sp_demand,expand_adj),axis=0)
        """
        expand_availdeg = self.available_degree[np.newaxis,:]
        obs = np.concatenate((self.demand,adj,expand_availdeg),axis=0)
        return obs

    def step(self, action):
        """
        :param action: [weights for nodes..., stop]
        :return: next_state, reward, done, {}
        """
        assert self.action_space.contains(action), "action type {} is invalid.".format(type(action))
        self.last_graph = copy.deepcopy(self.graph)

        """ original design
        # selected node pair
        ind = []
        n1 = node_weights.index(max(node_weights))
        ind.append(n1)
        node_weights[n1] = min(node_weights) - 0.1
        n2 = node_weights.index(max(node_weights))
        ind.append(n2)  

        # Take actions and Calculate intermediate reward
        if self.graph.has_edge(n1, n2):
            if self._check_connectivity(ind):
                self._remove_edge(ind)
                reward = self._cal_step_reward()
            else:  # Connectivity is disrupted
                reward = -self.connect_penalty
        else:
            if self._check_validity(ind):  # Valid edge on degree
                self._add_edge(ind)
                reward = self._cal_step_reward()
            else:   # Node degree violation
                reward = -self.degree_penalty
        """
        
        """design no.2
        # Start a new episode
        stop = action[-1] or self.counter >= self.max_action

        node_weights = action[:-1]
        node_weights = node_weights.tolist()
        
        n_max = node_weights.index(max(node_weights))
        n_min = node_weights.index(min(node_weights))
        node_weights = np.array(node_weights)
        add_ind = [n_max,n_min]

        # Check if both nodes have available degree
        if not self._check_degree(n_max):
            neighbors = [n for n in self.graph.neighbors(n_max)]
            weight_dif = abs(node_weights[neighbors] - node_weights[n_max])
            neighbors = np.array(neighbors)
            n_sim = np.where(weight_dif==np.min(weight_dif))[0].tolist()
            n_sim = neighbors[n_sim]
            for v in n_sim:
                rm_ind = [n_max,v]
                if self._check_connectivity(rm_ind):
                    self._remove_edge(rm_ind)
                    break
        if not self._check_degree(n_min):
            neighbors = [n for n in self.graph.neighbors(n_min)]
            weight_dif = abs(node_weights[neighbors] - node_weights[n_min])
            neighbors = np.array(neighbors)
            n_sim = np.where(weight_dif==np.min(weight_dif))[0].tolist()
            n_sim = neighbors[n_sim]
            for v in n_sim:
                rm_ind = [n_min,v]
                if self._check_connectivity(rm_ind):
                    self._remove_edge(rm_ind)
                    break

        if self._check_validity(add_ind):
            self._add_edge(add_ind)
            reward = self._cal_step_reward()
        else:
            reward = -self.penalty
        """

        Bmat = action.reshape((self.max_node,self.max_node))
        self._graph2Pvec(Bmat)
        obj = Bmat - np.tile(self.Pvec,(self.max_node,1)) - np.tile(self.Pvec.reshape(self.max_node,1),(1,self.max_node))
        adj = np.array(nx.adjacency_matrix(self.graph).todense(), np.float32)
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

        # Check if both nodes have available degree
        v1 = add_ind[0]
        v2 = add_ind[1]

        stop = self.counter >= self.max_action

        if np.max(masked_obj) <= 0:
            stop = True

        if v1 in self.prev_action and v2 in self.prev_action:
            stop = True

        reward = 0
        if not stop:
            if not self._check_degree(v1):
                neighbors = [n for n in self.graph.neighbors(v1)]
                h_neightbor = [Bmat[v1,n] for n in neighbors]
                v_n = neighbors[h_neightbor.index(min(h_neightbor))]
                rm_ind = [v_n,v1]
                if self._check_connectivity(rm_ind):
                    self._remove_edge(rm_ind)
                    print("remove edge ({0},{1})".format(v_n,v1))
            if not self._check_degree(v2):
                neighbors = [n for n in self.graph.neighbors(v2)]
                h_neightbor = [Bmat[v2,n] for n in neighbors]
                v_n = neighbors[h_neightbor.index(min(h_neightbor))]
                rm_ind = [v_n,v2]
                if self._check_connectivity(rm_ind):
                    self._remove_edge(rm_ind)
                    print("remove edge ({0},{1})".format(v_n,v2))
            if self._check_validity(add_ind):
                self._add_edge(add_ind)
                print("add edge ({0},{1})".format(v1,v2))

        reward = self._cal_reward_against_permatch()

        print("[Step{0}][Action{1}][Reward{2}]".format(self.counter,add_ind,reward))
        
        """
        sp_demand = self._demand_matrix_extend()
        #enc_adj = self._adj_extend(adj)
        expand_adj = adj[np.newaxis,:,:]
        #expand_adj = np.tile(adj[np.newaxis, np.newaxis,:,:], (self.max_node,1,1,1))
        #obs = np.concatenate((sp_demand,expand_adj),axis=1)
        obs = np.concatenate((sp_demand,expand_adj),axis=0)
        """
        expand_availdeg = self.available_degree[np.newaxis,:]
        obs = np.concatenate((self.demand,adj,expand_availdeg),axis=0)

        self.counter += 1
        if stop:
            self.reset()

        return obs, reward, stop, {}

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

    def _cal_reward_against_permatch(self):
        nn_score = 0
        permatch_score = 0
        #last_adj = np.array(nx.adjacency_matrix(self.last_graph).todense())
        #permatch_new_adj = self.permatch_baseline.n_steps_matching(self.demand,last_adj,self.available_degree)
        permatch_new_graph = self.permatch_baseline.n_steps_matching(
            self.demand,self.last_graph,self.allowed_degree,1) # single step weighted matching

        for s, d in itertools.product(range(self.max_node), range(self.max_node)):
            try:
                permatch_path_length = float(nx.shortest_path_length(permatch_new_graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                permatch_path_length = float(self.max_node)

            try:
                nn_path_length = float(nx.shortest_path_length(self.graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                nn_path_length = float(self.max_node)

            permatch_score += permatch_path_length * self.demand[s][d]
            nn_score += nn_path_length * self.demand[s][d]
        
        permatch_score /= (sum(sum(self.demand)))
        nn_score /= (sum(sum(self.demand)))
        #last_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        #cur_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        return nn_score - permatch_score


    def _add_edge(self, action):
        """
        :param action: [first_node, second_node]
        """
        #id_matrix = edge_graph.cal_id_matrix(self.max_node)
        self.graph.add_edge(action[0], action[1])
        #self.edges.nodes[id_matrix[action[0]][action[1]]]['feature'][-1] = 1

        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        """
        for j in range(self.max_node):
            if j > action[0]:
                self.edges.nodes[id_matrix[action[0]][j]]['feature'][3] -= 1
            else:
                self.edges.nodes[id_matrix[action[0]][j]]['feature'][4] -= 1

        for i in range(self.max_node):
            if i < action[1]:
                self.edges.nodes[id_matrix[i][action[1]]]['feature'][3] -= 1
            else:
                self.edges.nodes[id_matrix[i][action[1]]]['feature'][4] -= 1 

        # consistency check
        assert self.edges.nodes[0]['feature'][3] == self.available_degree[0]
        for i in range(1, self.max_node):
            assert self.edges.nodes[i-1]['feature'][4] == self.available_degree[i]
        """

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
        """
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

    def _demand_matrix_extend(self):
        """Converting demand matrix into N x N x N x N sparse matrix
        """
        sparse_demand = np.zeros([self.max_node**2,self.max_node,self.max_node],np.float32)
        srcs, dsts = self.demand.nonzero()
        for i in range(len(srcs)):
            s = srcs[i]
            d = dsts[i]
            demand = self.demand[s,d]
            sparse_demand[s*self.max_node+d,d,:] -= demand
            sparse_demand[s*self.max_node+d,:,s] += demand
        return sparse_demand

    def _adj_extend(self, adj):
        """
        Encoding available degree into adjacency matrix
        """
        deg_diag = np.diag(self.available_degree)
        return adj + deg_diag

    def adaptive_horizon(self):
        if ((self.episode_num != 0)
            and (self.episode_num % self.episode_expand == 0 )
            and (self.max_action < self.max_horizon)):

        self.max_action *= 2