import pickle as pk
import numpy as np
import networkx as nx

# Set parameters
data_count = 10000
dataset = []
num_nodes = 8
max_degree = 4
allowed_degree = np.ones(num_nodes) * max_degree
file_name = '../../data/traindata/{0}_{1}_{2}_topo_train.pk3'.format(data_count, num_nodes, max_degree)

def random_topology_generator(num_node, allowed_degree, tries=5):
    p = np.random.randn()
    success, G = adjust_graph(num_node, p, allowed_degree)
    count_try = 1
    while not success:
        if count_try > tries:
            return False, None
        if p<0.3:
            p+=0.1
        success, G = adjust_graph(num_node, p, allowed_degree)
        count_try += 1
    return True, G
            

def adjust_graph(num_node, p, allowed_degree):
    if p < 0.3:
        G = nx.fast_gnp_random_graph(num_node,p)
    else:
        G = nx.gnp_random_graph(num_node,p)
    success, new_graph = remove_extra_edge(allowed_degree, G)
    if not success:
        return False, None
    else:
        return True, new_graph

def remove_extra_edge(allowed_degree, graph):
    degree_inuse = np.array(graph.degree())[:,-1]
    while any(degree_inuse>allowed_degree):
        v = np.where(degree_inuse>allowed_degree)[0][0]
        neighbors = [n for n in graph.neighbors(v)]
        neighbors_deg = allowed_degree[neighbors].tolist()
        if max(neighbors_deg) <= 1:
            return False, None
        ind = neighbors_deg.index(max(neighbors_deg))
        v2 = neighbors[ind]
        graph.remove_edge(v,v2)
        degree_inuse = np.array(graph.degree())[:,-1]
    if nx.is_connected(graph):
        return True, graph
    else:
        return False, None

dataset_size = 0
i=0
while i < data_count:
    success, graph = random_topology_generator(num_nodes,allowed_degree)
    if success:
        graph_dict = nx.to_dict_of_dicts(graph)
        dataset.append(graph_dict)
        print("[datasize {}]".format(dataset_size))
        i+=1
        dataset_size += 1
print("[total datasize {}]".format(dataset_size))

    
with open(file_name, 'wb') as f:
    pk.dump(dataset, f)

