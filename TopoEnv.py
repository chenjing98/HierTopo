import tensorflow as tf
import gym
import networkx as nx
import copy
import itertools
import math
import edge_graph
import numpy as np


class TopoEnv(gym.Env):

    def __init__(self, fpath='10M_8_3.0_const3.pk3'):
        with open(fpath, 'rb') as f:
            self.dataset = pk.load(f)

        self.penalty = 1
        self.degree_penalty = 10
        self.max_action = 50

        self.trace_index = np.random.randint(len(dataset))
        self.demand = dataset[self.trace_index]['demand']
        self.allowed_degree = dataset[self.trace_index]['allowed_degree']
        self.available_degree = self.allowed_degree
        self.max_node = self.demand.shape[0]
        self.edge_graph = edge_graph.convert(demand=self.demand, allowed_degree=self.allowed_degree)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.max_node))
        self.counter = 0

        assert self.demand.shape[0] == self.demand.shape[1]

    def step(self, action):
        """
        :param action: [first_node, second_node, stop]
        :return: 
        """
        self.last_graph = copy.deepcopy(self.graph)

        # Start a new episode
        stop = action[2] or self.counter >= self.max_action

        # Take actions and Calculate intermediate reward
        if self._check_validity(action):    # Valid edge on degree
            if self._add_edge(action):  # Successfully added
                reward = self._cal_step_reward()
            else:   # Redundant edges
                reward = -self.penalty
        else:   # Node degree violation
            reward = -self.degree_penalty
        
        self.counter += 1
        if stop:
            self.trace_index = np.random.randint(len(dataset))
            self.demand = dataset[self.trace_index]['demand']
            self.allowed_degree = dataset[self.trace_index]['allowed_degree']
            self.available_degree = self.allowed_degree
            self.max_node = self.demand.shape[0]
            self.edge_graph = edge_graph.convert(demand=self.demand, allowed_degree=self.allowed_degree)
            self.graph = nx.Graph()
            self.graph.add_nodes_from(range(max_node))
            self.counter = 0

        return self.edge_graph, reward, stop
        
    def get_expert(self):
        pass
    
    def _cal_step_reward(self):
        last_score = 0
        cur_score = 0
        for s, d in itertools.product(range(self.max_node), range(self.max_node)):
            try:
                last_path_length = np.array(nx.shortest_path_length(self.last_graph))
            except nx.exception.NetworkXNoPath:
                last_path_length = self.max_node

            try:
                cur_path_length = np.array(nx.shortest_path_length(self.graph))
            except nx.exception.NetworkXNoPath:
                cur_path_length = self.max_node

            last_score += last_path_length * self.demand[s][d]
            cur_score += cur_path_length * self.demand[s][d]
        
        last_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        cur_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        return last_score - cur_score


    def _add_edge(self, action):
        """
        :param action: [first_node, second_node, stop]
        :return: success or not
        """
        if self.graph.has_edge(action[0], action[1]) or action[0] == action[1]:
            return False
        else:
            id_matrix = edge_graph.cal_id_matrix(self.max_node)
            self.graph.add_edge(action[0], action[1])
            self.edge_graph.nodes[id_matrix[action[0]][action[1]]]['feature'][-1] = 1

            # Update available degree
            self.available_degree = self.allowed_degree - self.graph.degree
            for j in range(self.max_node):
                if j > action[0]:
                    self.edge_graph.nodes[id_matrix[action[0]][j]]['feature'][3] -= 1
                else:
                    self.edge_graph.nodes[id_matrix[action[0]][j]]['feature'][4] -= 1

            for i in range(self.max_node):
                if i < action[1]:
                    self.edge_graph.nodes[id_matrix[i][action[1]]]['feature'][3] -= 1
                else:
                    self.edge_graph.nodes[id_matrix[i][action[1]]]['feature'][4] -= 1 

            # consistency check
            assert self.edge_graph.nodes[0]['feature'][3] == self.available_degree[0]
            for i in range(1, self.max_node):
                assert self.edge_graph.nodes[i-1]['feature'][4] == self.available_degree[i]

            return True

    def _check_validity(self, action):
        if self.available_degree(action[0]) < 1 or self.available_degree(action[1]) < 1:
            return False
        else:
            return True
