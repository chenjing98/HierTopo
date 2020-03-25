import tensorflow as tf
import gym
import networkx as nx
import copy
import itertools
import math
import numpy as np
import pickle as pk
from gym import spaces

class TopoEnv(gym.Env):

    def __init__(self, n_node=8, fpath='10M_8_3.0_const3.pk3'):
        with open(fpath, 'rb') as f:
            self.dataset = pk.load(f)

        self.connect_penalty = 0.05
        self.degree_penalty = 0.05
        self.max_action = 50
        self.max_node = n_node

        # isolated random number generator
        self.np_random = np.random.RandomState()

        self.reset()

        # define action space and observation space
        self.action_space = spaces.Box(low=0.0,high=1.0,shape=(self.max_node+1,)) #node weights & stop sign
        self.observation_space = spaces.Box(
            low=0.0,high=np.float32(1e6),shape=(self.max_node,self.max_node+1)) #featured adjacent matrix & available degrees

    def reset(self):
        self.counter = 0
        self.trace_index = np.random.randint(len(self.dataset))
        self.demand = self.dataset[self.trace_index]['demand']
        self.allowed_degree = self.dataset[self.trace_index]['allowed_degree']
        assert self.demand.shape[0] == self.demand.shape[1] # of shape [N,N]
        assert self.max_node == self.demand.shape[0], "Expect demand matrices of dimensions {0} but got {1}"\
            .format(self.max_node, self.demand.shape[0])

        self.available_degree = self.allowed_degree
        #self.edges = edge_graph.convert(demand=self.demand, allowed_degree=self.allowed_degree)
        
        # Initialize a path graph
        self.graph = nx.path_graph(self.max_node)

        E_adj = self._graph2mat()
        obs = np.concatenate((E_adj,self.available_degree[:,np.newaxis]),axis=-1)
        return obs

    def step(self, action):
        """
        :param action: [weights for nodes..., stop]
        :return: next_state, reward, done
        """
        assert self.action_space.contains(action), "action type {} is invalid.".format(type(action))
        self.last_graph = copy.deepcopy(self.graph)

        # Start a new episode
        stop = action[-1] or self.counter >= self.max_action

        node_weights = action[:-1]
        node_weights = node_weights.tolist()
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
        
        print("[Step: {0}][Reward: {1}]".format(self.counter, reward))
        # Update E_adj
        E_adj = self._graph2mat()
        obs = np.concatenate((E_adj,self.available_degree[:,np.newaxis]),axis=-1)
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
        
        last_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        cur_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        return last_score - cur_score


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

    def _check_connectivity(self, action):
        """
        Checking whether all (src, dst) pairs are stilled connected after removing the selected link.
        """
        self.graph.remove_edge(action[0],action[1])
        connected = True
        srcs, dsts = self.demand.nonzero()
        for i in range(len(srcs)):
            if not nx.has_path(self.graph,srcs[i],dsts[i]):
                connected = False
                break
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