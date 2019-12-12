import pickle as pk
import numpy as np
import gurobipy as gp
import edge_graph
import networkx as nx
import itertools
import time
import copy
import random

fpath = '10M_8_3.0_const3.pk3'
with open(fpath, 'rb') as f:
    dataset = pk.load(f)

match_objs = np.zeros(len(dataset))
match_time = np.zeros(len(dataset))
opt_objs = np.zeros(len(dataset))
opt_time = np.zeros(len(dataset))

for idx in range(1, len(dataset)):
    demand = dataset[idx]['demand']
    allowed_degree = dataset[idx]['allowed_degree']
    n = np.shape(demand)[0]

    # <===< gurobi >===>
    # m = gp.Model('topo')

    # # variables
    # vars = []
    # for i in range(n):
    #     for j in range(i+1, n):
    #         e_id = edge_graph.edge_id(i, j, n)
    #         vars.append(m.addVar(vtype=gp.GRB.BINARY, name=str(e_id)))

    # # constraints
    # id_matrix = edge_graph.cal_id_matrix(n)
    # for i in range(n):
    #     lhs = 0
    #     for j in range(n):
    #         if j == i:
    #             continue
    #         lhs += vars[id_matrix[i][j]]
    #     m.addConstr(lhs <= allowed_degree[i], 'c'+str(i))

    # # objective

    # <===< matching algorithm >===>
    # This is the greedy search algorithm introduced in ProjecToR [SIGCOMM'16], 
    # specially designed for the dedicated topology searching.

    # t1 = time.time()
    # edge_list = np.zeros(int(n*(n-1)/2))
    # adj_mat = np.zeros((n, n))
    # loop = True
    # while True:
        
    #     step_obj = np.inf
    #     step_id = -1
    #     for edge_id in np.where(edge_list == 0)[0]:
    #         i, j = edge_graph.cal_node_id(edge_id, n)
    #         if int(np.sum(adj_mat[i])) >= allowed_degree[i] or int(np.sum(adj_mat[j])) >= allowed_degree[j]:
    #             continue
                
    #         adj_tmp = copy.deepcopy(adj_mat)
    #         adj_tmp[i][j] = 1
    #         adj_tmp[j][i] = 1

    #         graph = nx.Graph(adj_tmp)
    #         temp_obj = 0
    #         for s, d in itertools.product(range(n), range(n)):
    #             try:
    #                 path_len = nx.shortest_path_length(graph, source=s, target=d)
    #             except nx.exception.NetworkXNoPath:
    #                 path_len = n
    #             temp_obj += path_len * demand[s][d]

    #         if temp_obj < step_obj:
    #             step_obj = temp_obj
    #             step_id = edge_id
        
    #     # No available edges
    #     if step_id == -1:
    #         break
        
    #     i, j = edge_graph.cal_node_id(step_id, n)
    #     edge_list[step_id] = 1
    #     adj_mat[i][j] = 1
    #     adj_mat[j][i] = 1

    # graph = nx.Graph(adj_mat)
    # match_obj = 0
    # for s, d in itertools.product(range(n), range(n)):
    #     try:
    #         path_len = nx.shortest_path_length(graph, source=s, target=d)
    #     except nx.exception.NetworkXNoPath:
    #         print(s, d, 'No Path')
    #         path_len = n
    #     match_obj += path_len * demand[s][d]
    
    # t2 = time.time()
    # match_objs[idx] = match_obj
    # match_time[idx] = t2 - t1
    # print(match_obj)


    # <===< search (enumeration) >===>
    
    mc_count = int(1e7)

    t1 = time.time()
    opt_obj = np.inf
    # for comb_id in range(2 ** (int(n*(n-1)/2))):
    for _ in range(mc_count):
        comb_id = random.randint(0, 2 ** (int(n*(n-1)/2)) - 1)
        edge_list = [comb_id >> d & 1 for d in range(int(n*(n-1)/2))][::-1]
        adj_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                adj_mat[i][j] = edge_list[edge_graph.edge_id(i, j, n)]
                adj_mat[j][i] = edge_list[edge_graph.edge_id(i, j, n)]

        if sum(np.sum(adj_mat, axis=0) > allowed_degree) > 0:
            continue

        graph = nx.Graph(adj_mat)
        temp_obj = 0
        for s, d in itertools.product(range(n), range(n)):
            try:
                path_len = nx.shortest_path_length(graph, source=s, target=d)
            except nx.exception.NetworkXNoPath:
                path_len = n
            temp_obj += path_len * demand[s][d]

        if temp_obj < opt_obj:
            opt_obj = temp_obj
            print(opt_obj)

    t2 = time.time()
    opt_objs[idx] = opt_obj
    opt_time[idx] = t2 - t1
    print(t2 - t1)


# with open('match.pk3', 'wb') as f:
#     dataset = pk.dump({'objs': match_objs, 'time': match_time}, f)

# with open('opt.pk3', 'wb') as f:
#     dataset = pk.dump({'objs': opt_objs, 'time': opt_time}, f)