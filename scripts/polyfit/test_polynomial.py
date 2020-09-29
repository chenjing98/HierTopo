import pickle as pk
import itertools
import random
import numpy as np
import networkx as nx
import multiprocessing
from permatch import permatch
from timeit import default_timer as timer

k = 4
n_nodes = 11
n_nodes_param = 11
n_iters = 3
degree_lim = 4
desired_output = 0.99
parallelism = 10
n_steps = 4

max_steps = int(n_nodes*degree_lim/2)
max_adjust_steps = 20

data_source = "scratch"

if data_source == "random":
    node_num = 8
    degree_lim = 4
    n_testings = 1000
    file_demand_degree = '../../data/10000_8_4_test.pk3'
    file_topo = "../../data/10000_8_4_topo_test.pk3"
elif data_source == "nsfnet":
    node_num = 14
    degree_lim = 4
    n_testings = 100
    file_demand_degree = '../../data/nsfnet/demand_100.pkl'
    file_topo = '../../data/nsfnet/topology.pkl'
elif data_source == "geant2":
    node_num = 24
    degree_lim = 8
    n_testings = 100
    file_demand_degree = '../../data/geant2/demand_100.pkl'
    file_topo = '../../data/geant2/topology.pkl'
elif data_source == "scratch":
    node_num = n_nodes
    n_testings = 1000
    #n_iters = int(n_nodes*(n_nodes-1)/2)
    file_demand = '../../data/10000_{0}_{1}_logistic.pk3'.format(n_nodes, degree_lim)
else:
    print("data_source {} unrecognized.".format(data_source))
    exit(1)

file_logging = '../../poly_log/log{0}_{1}_{2}_{3}.pkl'.format(n_nodes_param,degree_lim,k,n_iters)
with open(file_demand, 'rb') as f1:
    dataset = pk.load(f1)
#with open(file_topo, 'rb') as f2:
#    dataset_topo = pk.load(f2)
with open(file_logging, 'rb') as f3:
    solution = pk.load(f3)["solution"]
permatch_model = permatch(node_num)

def expand_orders_mat(feature):
    """
    :param feature: (np.array) N x N
    :return exp_feature: (np.array) N x N x k
    """
    N = feature.shape[0]
    exp_feature = np.zeros((N,N,k), np.float32)
    for i in range(k):
        exp_feature[:, :, i] = np.power(feature, i)
    return exp_feature

def cal_diff(v):
    N = v.shape[0]
    dif = np.repeat(np.expand_dims(v,0),N,0) - np.repeat(np.expand_dims(v,-1),N,-1)
    dif = np.abs(dif)
    return dif
    

def cal_pathlength(demand, graph):
    n_nodes = demand.shape[0]
    score = 0
    for s, d in itertools.product(range(n_nodes), range(n_nodes)):
        try:
            cur_path_length = float(nx.shortest_path_length(graph,source=s,target=d))
        except nx.exception.NetworkXNoPath:
            cur_path_length = float(n_nodes)

        score += cur_path_length * demand[s,d]
    score /= (sum(sum(demand)))
    return score

def apply_policy(demand, alpha):
    """
    :param demand: (np.array) N x N
    :param topo: (nx.dict_of_dicts)
    :param alpha: (np.array) N
    :return: metric: (np.float32) average shortest path length
    """

    n_nodes = node_num
    graph = nx.Graph()
    graph.add_nodes_from(list(range(n_nodes)))
    adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
    degree = np.sum(adj, axis=-1)

    z = np.zeros((n_nodes,n_nodes), np.float32)
    for _ in range(max_steps):
        #x = np.sum(demand, axis=0)
        x = demand/np.max(demand)*2 - 1 # [N]
        x = x.T
        for i in range(n_iters):
            exp_x = expand_orders_mat(x)
            weighing_self = np.matmul(exp_x, alpha[2*i*k:(2*i+1)*k])
            weighing_neigh = np.matmul(exp_x, alpha[(2*i+1)*k:(2*i+2)*k])
            neighbor_aggr = np.matmul(weighing_neigh, adj)
            g = weighing_self + neighbor_aggr
            #x = g/np.max(g)*2 # N x N
            gpos = np.where(g>=0,g,z)
            gneg = np.where(g<0,g,z)
            x = 1/(1+np.exp(-gpos)) + np.exp(gneg)/(1+np.exp(gneg)) - 1/2
        
        v = np.sum(x, axis=0)
        dif = cal_diff(v) + 1.0
        degree_full = np.where(degree>=degree_lim, 1.0, 0.0)
        degree_mask = np.repeat(np.expand_dims(degree_full,0),n_nodes,0) + np.repeat(np.expand_dims(degree_full,-1),n_nodes,-1)
        mask = adj + np.identity(n_nodes, np.float32) + degree_mask
        masked_dif = (mask == 0) * dif - 1.0
        ind_x, ind_y = np.where(masked_dif==np.max(masked_dif))
        #ind_x, ind_y = np.where(dif==np.max(dif))
        if len(ind_x) < 1:
            continue
        elif len(ind_x) > 1:
            j = random.randint(0, len(ind_x)-1)
            add_ind = (ind_x[j], ind_y[j])
        else:
            add_ind = (ind_x[0], ind_y[0])

        if (adj[add_ind] != 1) and (degree[add_ind[0]] < degree_lim) and (degree[add_ind[1]] < degree_lim):
            graph.add_edge(add_ind[0], add_ind[1])
            adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
            degree = np.sum(adj, axis=-1)
    
    path_length = cal_pathlength(demand, graph)
    return path_length

def apply_policy_replace(demand, alpha):
    """Policy with greedy initialization & link tearing down.
    :param demand: (np.array) N x N
    :param alpha: (np.array) N
    :return: metric: (np.float32) average shortest path length
    """

    n_nodes = node_num
    #graph = nx.Graph()
    #graph.add_nodes_from(list(range(n_nodes)))
    #adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
    adj = permatch_model.matching(demand, np.ones((node_num,)) * (degree_lim-1))
    graph = nx.from_numpy_matrix(adj)
    degree = np.sum(adj, axis=-1)

    z = np.zeros((n_nodes,n_nodes), np.float32)
    for s in range(max_adjust_steps):
        x = demand/np.max(demand)*2 - 1 # [N]
        x = x.T
        for i in range(n_iters):
            exp_x = expand_orders_mat(x)
            weighing_self = np.matmul(exp_x, alpha[2*i*k:(2*i+1)*k])
            weighing_neigh = np.matmul(exp_x, alpha[(2*i+1)*k:(2*i+2)*k])
            neighbor_aggr = np.matmul(weighing_neigh, adj)
            g = weighing_self + neighbor_aggr
            #x = g/np.max(g)*2 # N x N
            gpos = np.where(g>=0,g,z)
            gneg = np.where(g<0,g,z)
            x = 1/(1+np.exp(-gpos)) + np.exp(gneg)/(1+np.exp(gneg)) - 1/2
        
        v = np.sum(x, axis=0)
        dif = cal_diff(v) + 1.0
        mask = adj + np.identity(n_nodes, np.float32)
        masked_dif = (mask == 0) * dif - 1.0
        ind_x, ind_y = np.where(masked_dif==np.max(masked_dif))
        if len(ind_x) < 1:
            continue
        elif len(ind_x) > 1:
            j = random.randint(0, len(ind_x)-1)
            add_ind = (ind_x[j], ind_y[j])
        else:
            add_ind = (ind_x[0], ind_y[0])
        #if add_ind[0] == add_ind[1] or adj[add_ind] == 1:
        #    print("wrong in the find")

        rm_inds = []
        loss = 0
        if (degree[add_ind[0]] >= degree_lim):
            dif_at_n0 = np.max(dif) + 1.0 - dif[add_ind[0]]
            dif_n0_masked = np.multiply(adj[add_ind[0]],dif_at_n0)
            loss += np.max(dif) + 1.0 - np.max(dif_n0_masked)
            if loss > np.max(masked_dif):
                #print("Stop at No.{} step".format(s))
                break
            rm_ind = np.where(dif_n0_masked==np.max(dif_n0_masked))[0][0]
            #if not graph.has_edge(add_ind[0],rm_ind):
            #    print("wrong at first remove")
            graph.remove_edge(add_ind[0], rm_ind)
            rm_inds.append((add_ind[0], rm_ind))
        if (degree[add_ind[1]] >= degree_lim):
            dif_at_n1 = np.max(dif) + 1.0 - dif[add_ind[1]]
            dif_n1_masked = np.multiply(adj[add_ind[1]], dif_at_n1)
            loss += np.max(dif) + 1.0 - np.max(dif_n1_masked)
            if  loss > np.max(masked_dif):
                for removed in rm_inds:
                    graph.add_edge(removed[0],removed[1])
                #print("Stop at No.{} step".format(s))
                break
            rm_ind = np.where(dif_n1_masked==np.max(dif_n1_masked))[0][0]
            #if not graph.has_edge(add_ind[1],rm_ind):
            #    print("wrong at second remove")
            graph.remove_edge(add_ind[1], rm_ind)

        if (degree[add_ind[0]] < degree_lim) and (degree[add_ind[1]] < degree_lim):
            graph.add_edge(add_ind[0], add_ind[1])
        adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
        degree = np.sum(adj, axis=-1)
    
    if s == max_adjust_steps - 1:
        print("Unwillingly terminated.")    
    
    path_length = cal_pathlength(demand, graph)
    return path_length

def test_robust(solution, test_size):
    metrics = []
    for i in range(test_size):
        m = apply_policy(dataset[i], solution)
        metrics.append(m)
        #print("[No. {0}] {1}".format(i,m))
    output = np.mean(metrics)
    output_std = np.std(metrics)
    return output, output_std

t_begin = timer()
pred, pred_std = test_robust(solution, n_testings)
t_end = timer()
print("Prediction = {0}, std = {1}, test_time for {2} samples = {3}s".format(pred, pred_std, n_testings,t_end-t_begin))