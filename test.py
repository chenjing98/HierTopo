import math
import pickle as pk
import copy
from baseline_new.ego_tree_unit import *
from baseline_new.permatch import permatch
from baseline_new.create_file import create_file

import numpy as np
import itertools
import tensorflow as tf
from topoenv import TopoEnv

import networkx as nx
from whatisoptimal import optimal
from SLmodel import supervisedModel
#from h_shortest_path import TopoEnv,TopoOperator

MODEL_NAME = "model"
NUM_NODE = 8
ITERS = 10000
FOLDER = './data/'

def compute_reward(state, num_node, demand, degree,degree_penalty):    
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
    node_num = NUM_NODE
    folder = FOLDER
    env = TopoEnv(node_num)
    opt = optimal()

    scores_ego = []
    scores_match = []
    scores_nn = []
    scores_opt = []
    scores_greedy = []

    permatch_model = permatch(node_num)

    fpath = folder+'test'+str(node_num)+'_'+str(80)+'_'+str(4)+'.pk'
    with open(fpath, 'rb') as f1:
        dataset = pk.load(f1)

    with tf.Session() as sess:
        model = supervisedModel(sess, 8 ,4, [3,64,1])
        ckpt = tf.train.get_checkpoint_state('./')
        saver = tf.train.Saver()
        saver.restore(sess,ckpt.model_checkpoint_path)
        for i_iter in range(ITERS):
            demand = dataset[i_iter]['demand']
            degree = dataset[i_iter]['allowed_degree']       
            max_degree = max(degree)
            # test egotree
            test_e = ego_tree_unit(demand,node_num,degree,max_degree)
            test_e.create_tree()
            test_e.change_insert()
            state_e, _ = test_e.estab()
            score_e = compute_reward(state_e, node_num, demand, degree, 50)
            scores_ego.append(score_e)

            # test per-match
            state_m = permatch_model.matching(demand,degree)
            score_m = compute_reward(state_m, node_num, demand, degree, 100)
            scores_match.append(score_m)

            
            # test GNN+RL
            origin_obs = env.reset(demand=demand,degree=degree,provide=True)
            obs = copy.deepcopy(origin_obs)
            demand_input, adj_input, deg_input = obs_process(obs, node_num)
            potential = model.predict(demand_input,adj_input, deg_input)
            action = np.squeeze(potential)
            obs, _, _, _ = env.step(action)
            state_n = obs2adj(obs,node_num)
            print(state_n)
            score_n = compute_reward(state_n, node_num, demand, degree, 100)
            scores_nn.append(score_n)

            adj = obs2adj(origin_obs,node_num)
            origin_graph = nx.from_numpy_matrix(adj)
            _,_,opt_graph = opt.compute_optimal(node_num,origin_graph,demand,degree)
            state_o = np.array(nx.adjacency_matrix(opt_graph).todense(), np.float32)
            score_o = compute_reward(state_o, node_num, demand, degree, 100)
            scores_opt.append(score_o)

            # 2-step weighted matching for comparison
            origin_graph = nx.from_numpy_matrix(adj)
            permatch_new_graph = permatch_model.n_steps_matching(
                demand,origin_graph,degree,1)
            state_2m = np.array(nx.adjacency_matrix(permatch_new_graph).todense(), np.float32)
            score_2m = compute_reward(state_2m, node_num, demand, degree, 50)
            scores_greedy.append(score_2m)

            print("[iter {0}][egotree:{1}][permatch:{2}][nn: {3}][1-opt: {4}][greedy: {5}]".
                    format(i_iter,score_e,score_m,score_n,score_o,score_2m))

        
    print("Avg_scores: egotree{0} permatch{1} nn{2} opt{3} greedy{4}".format(
        np.mean(scores_ego),np.mean(scores_match),np.mean(scores_nn),np.mean(scores_opt),np.mean(scores_greedy)))

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