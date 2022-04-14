import copy
import pickle as pk

import numpy as np
import itertools
import networkx as nx

from multiprocessing import Pool
from timeit import default_timer as timer

from baseline.permatch import permatch
from baseline.dijkstra_greedy import DijGreedyAlg
from param_search.OptSearch import TopoOperator, dict2dict, dict2nxdict
from param_search.plotv import TopoSimulator

methods = ["greedy", "dijgreedy"]
# methods = ["oblivious-opt"] # options: "optimal", "greedy", "egotree", "param-search", "rl", "bmatch", "optimal-mp", "oblivious"
data_source = "scratch"  # options: "random8", "nsfnet", "geant2", "germany", "scratch"
scheme = "complete"  # options: "complete", "bysteps"
Max_degree = 4
n_steps = 1
n_nodes = 50

# parameters for "search"
alpha_v = 1.2
alpha_i = 0.1

# parameters for supervised learning & reinforcement learning
dims = [3, 64, 1]
model_name_sl = "../saved_model/model"
model_name_rl = "../saved_model/gnn_ppo4topo1"

if data_source == "random8":
    node_num = 8
    n_iters = 1000
    file_demand_degree = '../data/10000_8_4_test.pk3'
    file_topo = "../data/10000_8_4_topo_test.pk3"
elif data_source == "nsfnet":
    node_num = 14
    n_iters = 100
    file_demand_degree = '../data/nsfnet/demand_100.pkl'
    file_topo = '../data/nsfnet/topology.pkl'
elif data_source == "geant2":
    node_num = 24
    n_iters = 100
    file_demand_degree = '../data/geant2/demand_100.pkl'
    file_topo = '../data/geant2/topology.pkl'
elif data_source == "germany":
    node_num = 50
    n_iters = 100
    file_demand_degree = '../data/germany/demand_100.pkl'
    file_topo = '../data/germany/topology.pkl'
elif data_source == "scratch":
    node_num = n_nodes
    n_iters = 1000
    file_demand = '../data/2000_{0}_{1}_logistic.pk3'.format(
        n_nodes, Max_degree)
else:
    print("data_source {} unrecognized.".format(data_source))
    exit(1)


def cal_pathlength(state, n_node, demand, degree):
    D = copy.deepcopy(state)
    graph = nx.from_numpy_matrix(D)
    cost = 0
    for s, d in itertools.product(range(n_node), range(n_node)):
        try:
            path_length = float(
                nx.shortest_path_length(graph, source=s, target=d))
        except nx.exception.NetworkXNoPath:
            path_length = float(n_node)

        cost += path_length * demand[s, d]

    cost /= (sum(sum(demand)))
    return cost

def main():
    costs_dij = []

    t_begin = timer()
    # initialize models
    if "greedy" in methods:
        permatch_model = permatch(node_num)
    if "dijgreedy" in methods:
        dijgreedy_model = DijGreedyAlg(node_num, Max_degree)

    # load dataset
    if data_source == "scratch":
        with open(file_demand, 'rb') as f:
            dataset = pk.load(f)
    else:
        with open(file_demand_degree, 'rb') as f1:
            dataset = pk.load(f1)
        with open(file_topo, 'rb') as f2:
            dataset_topo = pk.load(f2)

    # start testing
    for i_iter in range(n_iters):
        if data_source == "random8":
            demand = dataset[i_iter]['demand']
            degree = dataset[i_iter]['allowed_degree']
            topo = dataset_topo[i_iter]
        elif data_source == "scratch":
            demand = dataset[i_iter]
            degree = Max_degree * np.ones((node_num, ), dtype=np.float32)
        else:
            demand = dataset[i_iter]
            degree = Max_degree * np.ones((node_num, ), dtype=np.float32)
            topo = dataset_topo

        # print("[iter {}]".format(i_iter))

        if "dijgreedy" in methods:
            if scheme == "bysteps":
                origin_graph = nx.from_dict_of_dicts(topo)
                result_graph = dijgreedy_model.topo_nsteps(
                    demand, origin_graph, degree, n_steps)
                state_d = np.array(
                    nx.adjacency_matrix(result_graph).todense(), np.float32)
            if scheme == "complete":
                state_d = dijgreedy_model.topo_scratch(demand, degree)
            cost_d = cal_pathlength(state_d, node_num, demand, degree)
            costs_dij.append(cost_d)
            # print("dijkstra greedy: {}".format(cost_d))

    t_end = timer()

    print("Setting:\ndata source = {0}\nn_nodes     = {1}".format(
        data_source, node_num))
    if scheme == "bysteps":
        print("======== Avg_costs & std ({} step(s)) ========".format(n_steps))
    elif scheme == "complete":
        print("========== Avg_costs & std (compl) ===========")

    if "dijgreedy" in methods:
        print("dijkstra greedy  : {0}  std : {1}".format(
            np.mean(costs_dij), np.std(costs_dij)))
    print("testing time : {} s".format(t_end - t_begin))


def obs2adj(obs, node_num):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    obs = np.reshape(obs, ((2 * node_num + 1), node_num))
    adj = obs[node_num:-1, :]
    #adj[adj>0] = 1
    return adj


def obs_process(obs, node_num):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    obs = np.reshape(obs, ((2 * node_num + 1), node_num))
    demand = obs[:node_num, :]
    adj = obs[node_num:-1, :]
    deg = obs[-1, :]
    #adj[adj>0] = 1
    return demand[np.newaxis, :], adj[np.newaxis, :], deg[np.newaxis, :]


if __name__ == "__main__":
    main()