import numpy as np
import networkx as nx
import copy
import itertools
import math
import random
import numpy as np
import pickle as pk
from plotv import TopoSimulator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start", type=int, default=0, help="the starting number for file naming")
args = parser.parse_args()

class TopoOperator(object):
    def __init__(self, n_node, max_iterstep=50, infinity=1e6, eps=1e-4):
        self.max_node = n_node
        self.inf = infinity
        self.eps = eps
        self.max_iterstep = max_iterstep
        self.steps = 4
        self.simulator = TopoSimulator(n_node=n_node)
        self.allowed_degree = 4 * np.ones((n_node, ))

    def reset(self, alpha_v, alpha_i, n_node=None):
        if n_node is not None:
            self.max_node = n_node
        self.alpha_v = alpha_v
        self.alpha_i = alpha_i

    def iter_v(self, V_neighbors, I_neighbors, R_neighbors):
        """ Updating V_i.
        :param V_neighbors: (list) voltages of i's neighbors
        :param I_neighbors: (list) I_ij for all i's neighbor node j
        :param R_neighbors: (list) R_ij for all i's neighbor node j
        :return: V_new: (float) the updated voltage of node i
        """
        expsum = 0
        for j in range(len(V_neighbors)):
            x_j = V_neighbors[j] + I_neighbors[j] * R_neighbors[j]
            expsum += abs(x_j) ** self.alpha_v * np.sign(x_j)
        expsum /= len(V_neighbors)
        V_new = abs(expsum) ** (1/self.alpha_v) * np.sign(expsum)
        return V_new

    def iter_i(self, I_ki, I_jk):
        """ Updating I_ij.
        :param I_ki: (list) I_ki for all i's neighbor node k
        :param I_jk: (list) I_jk for all j's neighbor node k
        :return: I_ij_new: (float) the updated current of link ij
        """
        expsum_i = 0
        expsum_j = 0
        for k in range(len(I_ki)):
            expsum_i += abs(I_ki[k]) ** self.alpha_i * np.sign(I_ki[k])
        expsum_i /= len(I_ki)
        for k in range(len(I_jk)):
            expsum_j += abs(I_jk[k]) ** self.alpha_i * np.sign(I_jk[k])
        expsum_j /= len(I_jk)
        I_ij_new = abs(expsum_i) ** (1/self.alpha_i) * np.sign(expsum_i) + \
                   abs(expsum_j) ** (1/self.alpha_i) * np.sign(expsum_j)
        I_ij_new /= 2
        return I_ij_new

    def init_features(self, graph):
        """Initializing Vs, Is and Rs.
        :param graph: (dict) nodes: [neighbor nodes]
        :return: self.V: (list)
        :return: self.I: (dict)
        :return: self.R: (dict)
        """
        self.V = [0 for _ in range(self.max_node)]
        self.I = dict()
        self.R = dict()
        for i, n in graph.items():
            self.I[i] = [0 for _ in range(len(n))]
            self.R[i] = [1 for _ in range(len(n))]

    def solve(self, graph, source, destination, demand):
        """
        :param graph: (dict) nodes as keys and their neighbors (put in a list) as values
        :param source: (int) the source node
        :param destination: (int) the destination node
        :param demand: (float) the traffic demand of the flow from source to destination
        :return: the converged V
        """
        iter_step = 0
        while iter_step < self.max_iterstep:
            # update V
            newV = []
            dif = 0
            for i in range(self.max_node):
                if i == destination:
                    V_i_new = 0.0
                else:
                    V_neighbors = []
                    for j in graph[i]:
                        V_neighbors.append(self.V[j])
                    V_i_new = self.iter_v(V_neighbors, self.I[i], self.R[i])
                newV.append(V_i_new)
                dif += (V_i_new - self.V[i]) ** 2
            # Has it converged?
            #if iter_step != 0  and dif < self.eps:
                #print("Converge step {}".format(iter_step))
                #break
            self.V = newV

            # update I
            newI = dict()
            for i in range(self.max_node):
                newI[i] = []
                I_ik = np.array(self.I[i])
                I_ki = - I_ik
                I_ki = I_ki.tolist()
                if i == source:
                    I_ki.append(demand)
                if i == destination:
                    I_ki.append(-demand)
                for j in graph[i]:
                    I_jk = copy.deepcopy(self.I[j])
                    if j == source:
                        I_jk.append(-demand)
                    if j == destination:
                        I_jk.append(demand)
                    V_ij_new = self.iter_i(I_ki,I_jk)
                    newI[i].append(V_ij_new)
            self.I = newI

            iter_step += 1

        #print("V {}".format(newV))
        return newV
    
    def run_onestep(self, graph, demand, no):
        """
        :param graph: (dict)
        :param demand: (matrix) a demand matrix
        """
        alpha_range = np.linspace(start=0, stop=2, num = 20)
        alpha_range = alpha_range.tolist()
        v_scans = []
        for alpha_v in alpha_range:
            for alpha_i in alpha_range:
                # scan for best alpha_v and alpha_i
                if alpha_v == 0 or alpha_i == 0:
                    continue
                self.reset(alpha_v, alpha_i)
                srcs, dsts = demand.nonzero()
                v = np.zeros((self.max_node,))
                for i in range(len(srcs)):
                    self.init_features(graph)
                    v_i = self.solve(graph, srcs[i], dsts[i], demand[srcs[i], dsts[i]])
                    v += np.array(v_i)
                v /= len(srcs)
                v_scans.append({'alpha_v':alpha_v,'alpha_i':alpha_i,'v':v})
        file_name = './vtest/'+'vscans_'+str(no)+'.pk'
        with open(file_name, 'wb') as f:
            pk.dump(v_scans, f)
    
    def run_multistep(self, graph, demand, no, n_steps):
        """
        :param graph: (dict)
        :param demand: (matrix) a demand matrix
        :param no: (int) a sequential number for file name
        :param n_steps: (int) how many steps to run
        """
        alpha_range = np.linspace(start=0, stop=3, num = 31)
        alpha_range = alpha_range.tolist()
        v_scans = []
        for alpha_v in alpha_range:
            for alpha_i in alpha_range:
                # scan for best alpha_v and alpha_i
                if alpha_v == 0 or alpha_i == 0:
                    continue
                self.reset(alpha_v, alpha_i)
                srcs, dsts = demand.nonzero()
                curr_graph = graph
                for i_step in range(n_steps):
                    v = np.zeros((self.max_node,))
                    for i in range(len(srcs)):
                        self.init_features(curr_graph)
                        v_i = self.solve(curr_graph, srcs[i], dsts[i], demand[srcs[i], dsts[i]])
                        v += np.array(v_i)
                    v /= len(srcs)
                    print("v {}".format(v))
                    if i_step < n_steps - 1:
                        new_graph = self.simulator.step_graph(self.max_node, v, 
                                                              demand=demand, 
                                                              topology=dict2nxdict(self.max_node,curr_graph),
                                                              allowed_degree=self.allowed_degree)
                        curr_graph = dict2dict(self.max_node,new_graph)
                    else:
                        cost = self.simulator.step(self.max_node, v, 
                                                   demand=demand, 
                                                   topology=dict2nxdict(self.max_node,curr_graph), 
                                                   allowed_degree=self.allowed_degree)
                print("cost {}".format(cost))
                v_scans.append({'alpha_v':alpha_v,'alpha_i':alpha_i,'cost':cost})
        file_name = './search/'+'alpha_cost_'+str(no)+'.pk'
        with open(file_name, 'wb') as f:
            pk.dump(v_scans, f)

    def predict(self, topo, demand):
        graph = dict2dict(self.max_node,topo)
        srcs, dsts = demand.nonzero()
        v = np.zeros((self.max_node,))
        for i in range(len(srcs)):
            self.init_features(graph)
            v_i = self.solve(graph, srcs[i], dsts[i], demand[srcs[i], dsts[i]])
            v += np.array(v_i)
        v /= len(srcs)
        return v


def adj2dict(n_node, adj):
    graph_dict = dict()
    for i in range(n_node):
        graph_dict[i] = []
        for j  in range(n_node):
            if adj[i,j] > 0:
                graph_dict[i].append(j)
    return graph_dict

def dict2dict(n_node, dic):
    graph_dict = dict()
    for i in range(n_node):
        graph_dict[i] = []
        for j in dic[i]:
            graph_dict[i].append(j)
    return graph_dict

def dict2nxdict(n_node, dic):
    graph_dict = dict()
    for i in range(n_node):
        graph_dict[i] = {}
        for j in dic[i]:
            graph_dict[i][j] = {}
    return graph_dict

def main():
    opr = TopoOperator(4)
    with open('10000_4_3.pk3', 'rb') as f1:
        dataset = pk.load(f1)
    with open('10000_4_3_topo.pk3', 'rb') as f2:
        topo_dataset = pk.load(f2)

    start_no = args.start
    for i in range(125):
        print("======== No {} =======".format(i+start_no))
        demand = dataset[i+start_no]['demand']
        topo = topo_dataset[i+start_no]
        graph = dict2dict(4,topo)
        opr.run_multistep(graph, demand, i+start_no, 4)

if __name__ == "__main__":
    main()