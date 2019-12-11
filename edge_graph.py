import networkx as nx
import numpy as np
 

def convert(demand, allowed_degree):
    assert demand.shape[0] == demand.shape[1], 'Demand is not a square matrix.'
    assert demand.shape[0] == allowed_degree.shape[0], 'Dimensions mismatch.'
    n = demand.shape[0]
    edge_graph = nx.Graph()
    id_matrix = cal_id_matrix(n)

    for i in range(n):
        for j in range(i+1, n):
            edge_graph.add_node(id_matrix[i][j], feature=[demand[i][j] + demand[j][i], i, j, allowed_degree[i], allowed_degree[j], 0])
            for k in range(j):
                if k == i:
                    continue
                edge_graph.add_edge(id_matrix[i][k], id_matrix[i][j])
            for k in range(i):
                if k == j:
                    continue
                edge_graph.add_edge(id_matrix[k][j], id_matrix[i][j])
    return edge_graph

def edge_id(i, j, n):
    if i > j:
        tmp = i
        i = j
        j = tmp
    return i*(n-(i+1)/2) + j-i-1

def cal_id_matrix(n):
    id_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            id_matrix[i][j] = i*(n-(i+1)/2) + j-i-1
            id_matrix[j][i] = i*(n-(i+1)/2) + j-i-1
    return id_matrix

def get_features(edge_graph):
    features = []
    feature_list = list(edge_graph.nodes.data())
    for i in range(edge_graph.number_of_nodes):
        features.append(feature_list[i][1]['feature'])
    return np.array(features)

def cal_node_id(e_id, num_nodes):
    for i in range(num_nodes):
        if (i+1)*(num_nodes-(i+2)/2) - 1 < e_id:
            continue
        for j in range(i+1, num_nodes):
            if edge_id(i, j) == e_id:
                return i, j
    return None
