import numpy as np
import itertools

class bMatching(object):
    def __init__(self, n_nodes, max_degree):
        self.n_nodes = n_nodes
        self.h = np.zeros(int(n_nodes*(n_nodes-1)/2), np.int)
        self.M = np.zeros(int(n_nodes*(n_nodes-1)/2), np.int)
        self.threshold = 1
        self.b = max_degree
        self.T = np.ones(int(n_nodes*(n_nodes-1)/2), np.int) * self.threshold
    
    def reset(self):
        self.h = np.zeros(int(self.n_nodes*(self.n_nodes-1)/2), np.int)
        self.M = np.zeros(int(self.n_nodes*(self.n_nodes-1)/2), np.int)
        self.T = np.ones(int(self.n_nodes*(self.n_nodes-1)/2), np.int) * self.threshold

    def match(self, demand):
        self.reset()
        for s, d in itertools.product(range(self.n_nodes), range(self.n_nodes)):
            if demand[s,d] == 0.0:
                continue
            tau = self.edge_id(s,d)
            if self.M[tau] == 0:
                self.h[tau] += 1
                if self.h[tau] == self.T[tau]:
                    self.FixSaturation(s,tau)
                    self.FixSaturation(d,tau)
                    if self.h[tau] == self.T[tau]:
                        self.FixMatching(s)
                        self.FixMatching(d)
                        self.M[tau] = 1
        return self.convert2adj()

    def FixSaturation(self, w, tau):
        Ew = [tau]
        count = 0
        for v in range(self.n_nodes):
            if v == w:
                continue
            e = self.edge_id(v,w)
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
            e = self.edge_id(v,w)
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
            return int(v1*(-v1+2*self.n_nodes-3)/2 + v2 - 1)
        else:
            return int(v2*(-v2+2*self.n_nodes-3)/2 + v1 - 1)

    def convert2adj(self):
        adj = np.zeros((self.n_nodes, self.n_nodes), np.float32)
        for v1 in range(self.n_nodes-1):
            for v2 in range(v1+1,self.n_nodes):
                e = self.edge_id(v1,v2)
                if self.M[e] == 1:
                    adj[v1,v2] = 1
                    adj[v2,v1] = 1
        return adj 