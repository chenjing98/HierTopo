import copy
import itertools
import numpy as np
import networkx as nx
import pickle as pk

import sys
sys.path.append("..")
from baseline.ego_tree_unit import ego_tree_unit
from baseline.permatch import permatch
from whatisoptimal import optimal
from param_search.OptSearch import TopoOperator, dict2dict, dict2nxdict
from param_search.plotv import TopoSimulator

MODEL_NAME = "model"
ITERS = 1000
FOLDER = './data/'

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
    alpha_v = 1.2
    alpha_i = 0.1
    n_steps = 4
    file_demand_degree = './data/10000_8_4_test.pk3'
    file_topo = "./data/10000_8_4_topo_test.pk3"
    
    opt = optimal()
    #env = TopoEnv(node_num)
    #scores_ego = []
    #scores_match = []
    #scores_nn = []
    scores_opt = []
    scores_greedy = []
    scores_sum = []

    permatch_model = permatch(node_num)
    opr = TopoOperator(node_num)
    opr.reset(alpha_v,alpha_i)
    sim = TopoSimulator(n_node=node_num)

    #fpath = folder+'test'+str(node_num)+'_'+str(80)+'_'+str(4)+'.pk'
    with open(file_demand_degree, 'rb') as f1:
        dataset = pk.load(f1)
    
    with open(file_topo, 'rb') as f2:
        dataset_topo = pk.load(f2)

    for i_iter in range(ITERS):
        demand = dataset[i_iter]['demand']
        degree = dataset[i_iter]['allowed_degree']     
        #int_degree = degree.astype(int) 
        #max_degree = int(max(degree))
        topo = dataset_topo[i_iter]
        """
        # test egotree
        test_e = ego_tree_unit(demand,node_num,int_degree,max_degree)
        test_e.create_tree()
        test_e.change_insert()
        state_e, _ = test_e.estab()
        score_e = compute_reward(state_e, node_num, demand, degree, 50)
        scores_ego.append(score_e)

        
        # test per-match
        state_m = permatch_model.matching(demand,degree)
        score_m = compute_reward(state_m, node_num, demand, degree, 100)
        scores_match.append(score_m)
        """

        # optimal
        #origin_graph = nx.from_dict_of_dicts(topo)
        #best_action,neigh,opt_graph = opt.compute_optimal(node_num,origin_graph,demand,degree)
        #print("OPT: {0}, neighbor2rm {1}".format(best_action,neigh))
        #state_o1 = np.array(nx.adjacency_matrix(opt_graph).todense(), np.float32)
        #score_o1 = compute_reward(state_o1, node_num, demand, degree, 100)

        #opt_dict = opt.multistep_compute_optimal(node_num,topo,demand,degree,n_steps)
        cost, opt_dict = opt.multistep_DFS(node_num,topo,demand,degree,n_steps)
        opt_graph = nx.from_dict_of_dicts(opt_dict)
        state_o = np.array(nx.adjacency_matrix(opt_graph).todense(), np.float32)
        score_o = compute_reward(state_o, node_num, demand, degree, 100)
        scores_opt.append(score_o)
        
        #v_optimal = opt.consturct_v(best_action,neigh)

        # n-step weighted matching for comparison
        origin_graph = nx.from_dict_of_dicts(topo)
        permatch_new_graph = permatch_model.n_steps_matching(
                demand,origin_graph,degree,n_steps)
        state_2m = np.array(nx.adjacency_matrix(permatch_new_graph).todense(), np.float32)
        score_2m = compute_reward(state_2m, node_num, demand, degree, 100)
        scores_greedy.append(score_2m)

        # weighted_sum
        
        #v_weightedsum = opr.predict(topo,demand)
        #print("V_weighted_sum {} ".format(v_weightedsum))
        #score_s = sim.step(8,v_weightedsum,0,demand,topo,degree)
        curr_graph = topo
        for i_step in range(n_steps):
            v = opr.predict(curr_graph, demand)
            if i_step < n_steps - 1:
                curr_graph = sim.step_graph(node_num, v, 
                                           demand=demand, 
                                           topology=curr_graph,
                                           allowed_degree=degree)
            else:
                score_s = sim.step(node_num, v, 
                                   demand=demand, 
                                   topology=curr_graph, 
                                   allowed_degree=degree)
        scores_sum.append(score_s)
        print("[iter{0}][opt:{1}][greedy:{2}][search:{3}]".
                    format(i_iter,score_o,score_2m,score_s))

        
    print("Avg_scores for {0} steps: opt{1} greedy{2} search{3}".format(n_steps,
        np.mean(scores_opt),np.mean(scores_greedy),np.mean(scores_sum)))
    
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