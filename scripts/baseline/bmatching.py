import numpy as np
import networkx as nx


class bMatching(object):

    def __init__(self, n_nodes, max_degree):
        self.n_nodes = n_nodes
        self.b = max_degree
        self.threshold = 2
        self.n_flows = 1000

        self.h = np.zeros(int(n_nodes * (n_nodes - 1) / 2), np.int)
        self.M = np.zeros(int(n_nodes * (n_nodes - 1) / 2), np.int)
        self.T = np.ones(int(n_nodes *
                             (n_nodes - 1) / 2), np.int) * self.threshold

    def reset(self):
        self.h = np.zeros(int(self.n_nodes * (self.n_nodes - 1) / 2), np.int)
        self.M = np.zeros(int(self.n_nodes * (self.n_nodes - 1) / 2), np.int)
        self.T = np.ones(int(self.n_nodes *
                             (self.n_nodes - 1) / 2), np.int) * self.threshold

    def match(self, demand):
        self.reset()
        cdf, p = self.demand_cdf(demand)
        path_lengths = []
        for _ in range(self.n_flows):
            rand = np.random.uniform(0, 1)
            count = 0
            for i in range(len(p)):
                if rand >= p[i]:
                    count += 1
                else:
                    break
            s, d = cdf[count]
            tau = self.edge_id(s, d)
            if self.M[tau] == 0:
                self.h[tau] += 1
                if self.h[tau] == self.T[tau]:
                    self.FixSaturation(s, tau)
                    self.FixSaturation(d, tau)
                    if self.h[tau] == self.T[tau]:
                        self.FixMatching(s)
                        self.FixMatching(d)
                        self.M[tau] = 1
            # l = self.cal_pathlength(s, d)
            # path_lengths.append(l)
        # return np.mean(path_lengths)

    def FixSaturation(self, w, tau):
        Ew = [tau]
        count = 0
        for v in range(self.n_nodes):
            if v == w:
                continue
            e = self.edge_id(v, w)
            if e == tau:
                continue
            Ew.append(e)
            if self.h[e] == self.T[e]:
                count += 1
        if count >= self.b:
            for e in Ew:
                self.h[e] = 0

    def FixMatching(self, w):
        Ew_M_intersect = []
        options = []
        count = 0
        for v in range(self.n_nodes):
            if v == w:
                continue
            e = self.edge_id(v, w)
            if self.M[e] == 1:
                count += 1
                Ew_M_intersect.append(e)
                if self.h[e] < self.T[e]:
                    options.append(e)
        if count == self.b:
            if len(options) == 0:
                raise ValueError
            else:
                self.M[options[0]] = 0

    def edge_id(self, v1, v2):
        if v1 < v2:
            return int(v1 * (-v1 + 2 * self.n_nodes - 3) / 2 + v2 - 1)
        else:
            return int(v2 * (-v2 + 2 * self.n_nodes - 3) / 2 + v1 - 1)

    def convert2adj(self):
        adj = np.zeros((self.n_nodes, self.n_nodes), np.float32)
        for v1 in range(self.n_nodes - 1):
            for v2 in range(v1 + 1, self.n_nodes):
                e = self.edge_id(v1, v2)
                if self.M[e] == 1:
                    adj[v1, v2] = 1
                    adj[v2, v1] = 1
        return adj

    def cal_pathlength(self, s, d):
        adj = self.convert2adj()
        G = nx.from_numpy_matrix(adj)
        try:
            path_length = float(nx.shortest_path_length(G, source=s, target=d))
        except nx.exception.NetworkXNoPath:
            path_length = float(self.n_nodes)
        return path_length

    def demand_cdf(self, demand):
        demand_norm = demand / np.sum(demand)
        p = []
        p_accum = 0
        count = 0
        cdf = {}
        for v1 in range(self.n_nodes):
            for v2 in range(self.n_nodes):
                if v1 == v2 or demand[v1, v2] == 0:
                    continue
                cdf[count] = (v1, v2)
                count += 1
                p_accum += demand_norm[v1, v2]
                p.append(p_accum)
        return cdf, p
