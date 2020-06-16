import pickle as pk
import copy

import numpy as np
import itertools

import networkx as nx
from whatisoptimal import optimal
from OptSearch import TopoOperator, dict2dict, dict2nxdict
from plotv import TopoSimulator

MODEL_NAME = "model"
ITERS = 1000
FOLDER = './data/'
log_file = './logging_opt.pk3'

def compute_reward(state, num_node, demand, degree, degree_penalty):    
    D = copy.deepcopy(state)
    graph = nx.from_numpy_matrix(D)
    cost = 0
    for s, d in itertools.product(range(num_node), range(num_node)):
        try:
            path_length = float(nx.shortest_path_length(graph,source=s,target=d))
        except nx.exception.NetworkXNoPath:
            path_length = float(num_node)

        cost += path_length * demand[s,d]   

    cost /= (sum(sum(demand)))   
    return cost 

def main():
    # Set the parameters
    node_num = 8

    file_demand_degree = './data/10000_8_4_test.pk3'
    #file_topo = "./data/10000_8_4_topo_test.pk3"
    
    opt = optimal()
    scores_opt = []

    with open(file_demand_degree, 'rb') as f1:
        dataset = pk.load(f1)
    
    f2 = open(log_file, 'wb')

    for i_iter in range(ITERS):
        demand = dataset[i_iter]['demand']
        degree = dataset[i_iter]['allowed_degree']     

        #opt_dict = opt.multistep_compute_optimal(node_num,topo,demand,degree,n_steps)
        cost, opt_dict = opt.optimal_topology(node_num, demand, degree)
        score_o = cost/(sum(sum(demand)))   
        scores_opt.append(score_o)
        
        print("[iter{0}][opt:{1}]".
                    format(i_iter,score_o))
        if i_iter % 50 == 0:
            pk.dump(f2,[i_iter,np.mean(scores_opt)])
            print("Avg_scores at iter {0}: {1}".format(i_iter,np.mean(scores_opt)))
    
    print("Avg_scores: opt{0}".format(np.mean(scores_opt)))
    
    """
    np.savez("./dataset_{0}_{1}_dagger.npz".format(node_num,ITERS),
                v=collect_act,
                demand=collect_demand,
                adj=collect_adj,
                degree=collect_deg)
    """

def obs2adj(obs,node_num):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    obs = np.reshape(obs,((2*node_num+1),node_num))
    adj = obs[node_num:-1,:]
    #adj[adj>0] = 1
    return adj

def obs_process(obs,node_num):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    obs = np.reshape(obs,((2*node_num+1),node_num))
    demand = obs[:node_num,:]
    adj = obs[node_num:-1,:]
    deg = obs[-1, :]
    #adj[adj>0] = 1
    return demand[np.newaxis,:], adj[np.newaxis,:], deg[np.newaxis,:]

if __name__ == "__main__":
    main()