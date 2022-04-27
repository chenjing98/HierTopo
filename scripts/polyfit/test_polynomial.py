import copy
import pickle as pk

import numpy as np
import random
import itertools
import networkx as nx

from multiprocessing import Pool
from timeit import default_timer as timer

from permatch import permatch

k = 3
n_nodes = 30
n_nodes_param = 10
n_iters = 14
n_iters_param = 10
degree_lim = 4
NUM_PARALLEL = 10

MAX_STEPS = int(n_nodes*degree_lim/2)
MAX_STEPS_REAL = 20

adding_mode = "replace"

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
    file_demand = '../../data/2000_{0}_{1}_logistic.pk3'.format(n_nodes, degree_lim)
else:
    print("data_source {} unrecognized.".format(data_source))
    exit(1)

file_logging = '../../poly_log/log{0}_{1}_{2}_{3}_same.pkl'.format(n_nodes_param,degree_lim,k,n_iters_param)
if adding_mode == "replace":
    file_logging = '../../poly_log/log{0}_{1}_{2}_{3}_same_repl.pkl'.format(n_nodes_param,degree_lim,k,n_iters_param)
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
    #graph = nx.from_dict_of_dicts(dataset_topo)

    adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
    degree = np.sum(adj, axis=-1)

    z = np.zeros((n_nodes,n_nodes), np.float32)
    for _ in range(MAX_STEPS):
        #x = np.sum(demand, axis=0)
        x = demand/np.max(demand)*2 - 1 # [N]
        x = x.T
        for i in range(n_iters):
            exp_x = expand_orders_mat(x)
            weighing_self = np.matmul(exp_x, alpha[0:k])
            weighing_neigh = np.matmul(exp_x, alpha[k:2*k])
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
    for s in range(MAX_STEPS_REAL):
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
    
    if s == MAX_STEPS_REAL - 1:
        print("Unwillingly terminated.")    
    
    path_length = cal_pathlength(demand, graph)
    return path_length

def apply_policy_replace_nsquare_list(demand, alpha):
    graph = nx.Graph()
    graph.add_nodes_from(list(range(node_num)))
    adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
    #adj = permatch_model.matching(demand, np.ones((node_num,)) * (degree_lim-1))
    #graph = nx.from_numpy_matrix(adj)
    degree = np.sum(adj, axis=-1)
    
    remaining_choices = []
    for i in range(node_num-1):
        for j in range(i+1,node_num):
            remaining_choices.append(i*node_num+j)
    rm_inds = []
    failed_attempts = []

    v = cal_v(demand, alpha, adj)
    dif_e = cal_diff_inrange(v,remaining_choices)
    while remaining_choices:
        curr_e_num = dif_e.index(max(dif_e))
        curr_e = remaining_choices[curr_e_num]
        v1 = int(curr_e/node_num)
        v2 = curr_e % node_num
        if adj[v1,v2] == 1:
            del remaining_choices[curr_e_num]
            del dif_e[curr_e_num]
            continue
        if degree[v1] < degree_lim and degree[v2] < degree_lim:
            graph.add_edge(v1,v2)
            adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
            degree = np.sum(adj, axis=-1)
            v = cal_v(demand, alpha, adj)
            del remaining_choices[curr_e_num]
            dif_e = cal_diff_inrange(v,remaining_choices)
            continue
        if len(failed_attempts) > 20:
            del remaining_choices[curr_e_num]
            del dif_e[curr_e_num]
            continue
        # need to remove some edges
        if degree[v1] >= degree_lim and degree[v2] >= degree_lim:
            v1_neighbor = [n for n in graph.neighbors(v1)]
            v1_edges = np.where(np.array(v1_neighbor) > v1, v1 * node_num + np.array(v1_neighbor), np.array(v1_neighbor) * node_num + v1).tolist()
            dif_v1 = cal_diff_inrange(v, v1_edges)
            v1_e_num = dif_v1.index(min(dif_v1))
            e1_rm = v1_edges[v1_e_num]

            v2_neighbor = [n for n in graph.neighbors(v2)]
            v2_edges = np.where(np.array(v2_neighbor) > v2, v2 * node_num + np.array(v2_neighbor), np.array(v2_neighbor) * node_num + v2).tolist()
            dif_v2 = cal_diff_inrange(v, v2_edges)
            v2_e_num = dif_v2.index(min(dif_v2))
            e2_rm = v2_edges[v2_e_num]

            rm_inds = [e1_rm, e2_rm]
            adj_rp = copy.deepcopy(adj)
            adj_rp[int(e1_rm/node_num),e1_rm%node_num] = 0
            adj_rp[e1_rm%node_num,int(e1_rm/node_num)] = 0
            adj_rp[int(e2_rm/node_num),e2_rm%node_num] = 0
            adj_rp[e2_rm%node_num,int(e2_rm/node_num)] = 0
            adj_rp[v1,v2] = 1
            adj_rp[v2,v1] = 1
            v_rp = cal_v(demand, alpha, adj_rp)
            if max(dif_e) + sum(cal_diff_inrange(v,rm_inds)) > sum(cal_diff_inrange(v_rp,[curr_e])) + sum(cal_diff_inrange(v_rp, rm_inds)):
                graph.remove_edge(int(e1_rm/node_num),e1_rm%node_num)
                graph.remove_edge(int(e2_rm/node_num),e2_rm%node_num)
                graph.add_edge(v1,v2)
                adj = adj_rp
                degree = np.sum(adj, axis=-1)
                v = v_rp
                del remaining_choices[curr_e_num]
                dif_e = cal_diff_inrange(v,remaining_choices)
            else:
                failed_attempts.append(curr_e)
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
        elif degree[v1] >= degree_lim:
            v1_neighbor = [n for n in graph.neighbors(v1)]
            v1_edges = np.where(np.array(v1_neighbor) > v1, v1 * node_num + np.array(v1_neighbor), np.array(v1_neighbor) * node_num + v1).tolist()
            dif_v1 = cal_diff_inrange(v, v1_edges)
            v1_e_num = dif_v1.index(min(dif_v1))
            e1_rm = v1_edges[v1_e_num]
            rm_inds.append(e1_rm)
            adj_rp = copy.deepcopy(adj)
            adj_rp[int(e1_rm/node_num),e1_rm%node_num] = 0
            adj_rp[e1_rm%node_num,int(e1_rm/node_num)] = 0
            adj_rp[v1,v2] = 1
            adj_rp[v2,v1] = 1
            v_rp = cal_v(demand, alpha, adj_rp)
            if max(dif_e) + sum(cal_diff_inrange(v,rm_inds)) > sum(cal_diff_inrange(v_rp,[curr_e])) + sum(cal_diff_inrange(v_rp, rm_inds)):
                graph.remove_edge(int(e1_rm/node_num),e1_rm%node_num)
                graph.add_edge(v1,v2)
                adj = adj_rp
                degree = np.sum(adj, axis=-1)
                v = v_rp
                del remaining_choices[curr_e_num]
                dif_e = cal_diff_inrange(v,remaining_choices)
                rm_inds = []
            else:
                failed_attempts.append(curr_e)
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
        else:
            v2_neighbor = [n for n in graph.neighbors(v2)]
            v2_edges = np.where(np.array(v2_neighbor) > v2, v2 * node_num + np.array(v2_neighbor), np.array(v2_neighbor) * node_num + v2).tolist()
            dif_v2 = cal_diff_inrange(v, v2_edges)
            v2_e_num = dif_v2.index(min(dif_v2))
            e2_rm = v2_edges[v2_e_num]
            rm_inds.append(e2_rm)
            adj_rp = copy.deepcopy(adj)
            adj_rp[int(e2_rm/node_num),e2_rm%node_num] = 0
            adj_rp[e2_rm%node_num,int(e2_rm/node_num)] = 0
            adj_rp[v1,v2] = 1
            adj_rp[v2,v1] = 1
            v_rp = cal_v(demand, alpha, adj_rp)
            if max(dif_e) + sum(cal_diff_inrange(v,rm_inds)) > sum(cal_diff_inrange(v_rp,[curr_e])) + sum(cal_diff_inrange(v_rp, rm_inds)):
                graph.remove_edge(int(e2_rm/node_num),e2_rm%node_num)
                graph.add_edge(v1,v2)
                adj = adj_rp
                degree = np.sum(adj, axis=-1)
                v = v_rp
                del remaining_choices[curr_e_num]
                dif_e = cal_diff_inrange(v,remaining_choices)
                rm_inds = []
            else:
                failed_attempts.append(curr_e)
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
    #print(graph.number_of_edges())
    path_length = cal_pathlength(demand, graph)
    return path_length


def apply_policy_replace_run(params):
    demand = params["demand"]
    alpha = params["alpha"]
    graph = nx.Graph()
    graph.add_nodes_from(list(range(node_num)))
    adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
    #adj = permatch_model.matching(demand, np.ones((node_num,)) * (degree_lim-1))
    #graph = nx.from_numpy_matrix(adj)
    degree = np.sum(adj, axis=-1)
    
    remaining_choices = []
    for i in range(node_num-1):
        for j in range(i+1,node_num):
            remaining_choices.append(i*node_num+j)
    rm_inds = []
    failed_attempts = []

    v = cal_v(demand, alpha, adj)
    dif_e = cal_diff_inrange(v,remaining_choices)
    while remaining_choices:
        curr_e_num = dif_e.index(max(dif_e))
        curr_e = remaining_choices[curr_e_num]
        v1 = int(curr_e/node_num)
        v2 = curr_e % node_num
        if adj[v1,v2] == 1:
            del remaining_choices[curr_e_num]
            del dif_e[curr_e_num]
            continue
        if degree[v1] < degree_lim and degree[v2] < degree_lim:
            graph.add_edge(v1,v2)
            adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
            degree = np.sum(adj, axis=-1)
            v = cal_v(demand, alpha, adj)
            del remaining_choices[curr_e_num]
            dif_e = cal_diff_inrange(v,remaining_choices)
            continue
        if len(failed_attempts) > 20:
            del remaining_choices[curr_e_num]
            del dif_e[curr_e_num]
            continue
        # need to remove some edges
        if degree[v1] >= degree_lim and degree[v2] >= degree_lim:
            v1_neighbor = [n for n in graph.neighbors(v1)]
            v1_edges = np.where(np.array(v1_neighbor) > v1, v1 * node_num + np.array(v1_neighbor), np.array(v1_neighbor) * node_num + v1).tolist()
            dif_v1 = cal_diff_inrange(v, v1_edges)
            v1_e_num = dif_v1.index(min(dif_v1))
            e1_rm = v1_edges[v1_e_num]

            v2_neighbor = [n for n in graph.neighbors(v2)]
            v2_edges = np.where(np.array(v2_neighbor) > v2, v2 * node_num + np.array(v2_neighbor), np.array(v2_neighbor) * node_num + v2).tolist()
            dif_v2 = cal_diff_inrange(v, v2_edges)
            v2_e_num = dif_v2.index(min(dif_v2))
            e2_rm = v2_edges[v2_e_num]

            rm_inds = [e1_rm, e2_rm]
            adj_rp = copy.deepcopy(adj)
            adj_rp[int(e1_rm/node_num),e1_rm%node_num] = 0
            adj_rp[e1_rm%node_num,int(e1_rm/node_num)] = 0
            adj_rp[int(e2_rm/node_num),e2_rm%node_num] = 0
            adj_rp[e2_rm%node_num,int(e2_rm/node_num)] = 0
            adj_rp[v1,v2] = 1
            adj_rp[v2,v1] = 1
            v_rp = cal_v(demand, alpha, adj_rp)
            if max(dif_e) + sum(cal_diff_inrange(v,rm_inds)) > sum(cal_diff_inrange(v_rp,[curr_e])) + sum(cal_diff_inrange(v_rp, rm_inds)):
                graph.remove_edge(int(e1_rm/node_num),e1_rm%node_num)
                graph.remove_edge(int(e2_rm/node_num),e2_rm%node_num)
                graph.add_edge(v1,v2)
                adj = adj_rp
                degree = np.sum(adj, axis=-1)
                v = v_rp
                del remaining_choices[curr_e_num]
                dif_e = cal_diff_inrange(v,remaining_choices)
            else:
                failed_attempts.append(curr_e)
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
        elif degree[v1] >= degree_lim:
            v1_neighbor = [n for n in graph.neighbors(v1)]
            v1_edges = np.where(np.array(v1_neighbor) > v1, v1 * node_num + np.array(v1_neighbor), np.array(v1_neighbor) * node_num + v1).tolist()
            dif_v1 = cal_diff_inrange(v, v1_edges)
            v1_e_num = dif_v1.index(min(dif_v1))
            e1_rm = v1_edges[v1_e_num]
            rm_inds.append(e1_rm)
            adj_rp = copy.deepcopy(adj)
            adj_rp[int(e1_rm/node_num),e1_rm%node_num] = 0
            adj_rp[e1_rm%node_num,int(e1_rm/node_num)] = 0
            adj_rp[v1,v2] = 1
            adj_rp[v2,v1] = 1
            v_rp = cal_v(demand, alpha, adj_rp)
            if max(dif_e) + sum(cal_diff_inrange(v,rm_inds)) > sum(cal_diff_inrange(v_rp,[curr_e])) + sum(cal_diff_inrange(v_rp, rm_inds)):
                graph.remove_edge(int(e1_rm/node_num),e1_rm%node_num)
                graph.add_edge(v1,v2)
                adj = adj_rp
                degree = np.sum(adj, axis=-1)
                v = v_rp
                del remaining_choices[curr_e_num]
                dif_e = cal_diff_inrange(v,remaining_choices)
                rm_inds = []
            else:
                failed_attempts.append(curr_e)
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
        else:
            v2_neighbor = [n for n in graph.neighbors(v2)]
            v2_edges = np.where(np.array(v2_neighbor) > v2, v2 * node_num + np.array(v2_neighbor), np.array(v2_neighbor) * node_num + v2).tolist()
            dif_v2 = cal_diff_inrange(v, v2_edges)
            v2_e_num = dif_v2.index(min(dif_v2))
            e2_rm = v2_edges[v2_e_num]
            rm_inds.append(e2_rm)
            adj_rp = copy.deepcopy(adj)
            adj_rp[int(e2_rm/node_num),e2_rm%node_num] = 0
            adj_rp[e2_rm%node_num,int(e2_rm/node_num)] = 0
            adj_rp[v1,v2] = 1
            adj_rp[v2,v1] = 1
            v_rp = cal_v(demand, alpha, adj_rp)
            if max(dif_e) + sum(cal_diff_inrange(v,rm_inds)) > sum(cal_diff_inrange(v_rp,[curr_e])) + sum(cal_diff_inrange(v_rp, rm_inds)):
                graph.remove_edge(int(e2_rm/node_num),e2_rm%node_num)
                graph.add_edge(v1,v2)
                adj = adj_rp
                degree = np.sum(adj, axis=-1)
                v = v_rp
                del remaining_choices[curr_e_num]
                dif_e = cal_diff_inrange(v,remaining_choices)
                rm_inds = []
            else:
                failed_attempts.append(curr_e)
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
    #print(graph.number_of_edges())
    return graph


def cal_diff_inrange(v, edges):
    dif = []
    for i in range(len(edges)):
        e = edges[i]
        v1 = int(e/node_num)
        v2 = e % node_num
        dif.append(np.abs(v[v1]-v[v2]))
    return dif

def cal_v(demand, alpha, adj):
    x = demand/np.max(demand)*2 - 1 # [N]
    x = x.T
    z = np.zeros((node_num,node_num), np.float32)
    for i in range(n_iters):
        exp_x = expand_orders_mat(x)
        weighing_self = np.matmul(exp_x, alpha[0:k])
        weighing_neigh = np.matmul(exp_x, alpha[k:2*k])
        neighbor_aggr = np.matmul(weighing_neigh, adj)
        g = weighing_self + neighbor_aggr
        #x = g/np.max(g)*2 # N x N
        gpos = np.where(g>=0,g,z)
        gneg = np.where(g<0,g,z)
        x = 1/(1+np.exp(-gpos)) + np.exp(gneg)/(1+np.exp(gneg)) - 1/2
        
    v = np.sum(x, axis=0)
    return v

def test_robust(solution, test_size):
    metrics = []
    for i in range(test_size):
        if adding_mode == "add":
            m = apply_policy(dataset[i], solution)
        else:
            m = apply_policy_replace_nsquare_list(dataset[i], solution)
        metrics.append(m)
        #print("[No. {0}] {1}".format(i,m))
    output = np.mean(metrics)
    output_std = np.std(metrics)
    return output, output_std

def test_mp(solution, test_size):
    # Run the test parallelly
    params = []
    metrics = []
    t0 = timer()
    for i in range(test_size):
        param = {}
        param["demand"] = dataset[i]
        param["alpha"] = solution
        params.append(param)
    pool = Pool()
    graphs = pool.map(apply_policy_replace_run, params)
    pool.close()
    pool.join()
    t1 = timer()
    print("Decision Time {}".format(t1-t0))
    for i in range(test_size):
        m = cal_pathlength(dataset[i], graphs[i])
        metrics.append(m)
    output = np.mean(metrics)
    output_std = np.std(metrics)
    return output, output_std

"""
def test_oblivious(solution, test_size):
    metrics = []
    demand = np.zeros((n_nodes,n_nodes),np.float32)
    for i_iter in range(test_size):
        demand += dataset[i_iter]
    demand /= test_size
    graph = apply_policy_replace_graph(demand, solution)
    graph_adj = np.array(nx.adjacency_matrix(graph).todense(),np.int8)
    with open("./adj_oblivious.pkl","wb") as f:
        pk.dump(graph_adj,f)
    for i in range(test_size):
        m = cal_pathlength(dataset[i], graph)
        metrics.append(m)
        #print("[No. {0}] {1}".format(i,m))
    output = np.mean(metrics)
    output_std = np.std(metrics)
    return output, output_std
"""

t_begin = timer()
pred, pred_std = test_mp(solution, n_testings)
t_end = timer()
print("Prediction = {0}, std = {1}, test_time for {2} samples = {3}s".format(pred, pred_std, n_testings,t_end-t_begin))