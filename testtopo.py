import math
import pickle
import copy
from baseline_new.ego_tree_unit import *
from baseline_new.permatch import *
from baseline_new.create_file import create_file

import numpy as np
#import tensorflow as tf
#from stable_baselines import PPO2
#from topoenv import TopoEnv

import networkx as nx
from h_shortest_path import TopoEnv,TopoOperator

MODEL_NAME = "ppo4topo"
NUM_NODE = 8
COUNT = 8 # ?
ITERS = 1000
FOLDER = './data/'

def compute_reward(state, node_num, demand, degree,degree_penalty):    
    D = copy.deepcopy(state)

    # floyd shortest path algorithm
    for i in range(node_num):
        for j in range(node_num):
            if (D[i][j] == 0) & (i != j):
                D[i][j] = np.inf
    for k in range(node_num):
        for i in range(node_num):
            for j in range(node_num):
                if(D[i][j]>D[i][k]+D[k][j]):
                    D[i][j]=D[i][k]+D[k][j]
    # D[i,j] = L(path(i,j))
    # compute weighted path length
    score = 0
    for i in range(node_num-1):
        for j in range(i+1,node_num):
            if(D[i,j]) > node_num:
                score += demand[i,j]*node_num
            else:
                score += demand[i,j]*D[i,j] 
    
    # degree penalty
    for i in range(node_num):
        if np.sum(state[i,:]) > degree[i]:
            score = score + degree_penalty            
    return score 

def main():
    node_num = NUM_NODE
    count = COUNT
    folder = FOLDER
    
    #f = createfile(node_num,folder=folder)
    #f.create()

    #policy = PPO2.load(MODEL_NAME)
    #env = TopoEnv()
    opr = TopoOperator(node_num)
    env = TopoEnv(node_num)

    scores_ego = []
    scores_match = []
    #scores_nn = []
    scores_h = []
    steps_h = []

    for i_ter in range(ITERS):
        with open(folder+str(node_num)+'demand_'+str(i_ter)+'.pk',"rb") as fp:
            demand = pickle.load(fp)
        with open(folder+str(node_num)+'degree_'+str(i_ter)+'.pk',"rb") as fp:
            degree = pickle.load(fp)          
        max_degree = max(degree)
        # test egotree
        test_e = ego_tree_unit(demand,node_num,degree,max_degree)
        test_e.create_tree()
        test_e.change_insert()
        state_e, count = test_e.estab()
        score_e = compute_reward(state_e, node_num, demand, degree, 50)
        scores_ego.append(score_e)

        # test per-match
        test_m = permatch(demand,node_num,degree)
        state_m = test_m.matching(count)
        score_m = compute_reward(state_m, node_num, demand, degree, 50)
        scores_match.append(score_m)

        """
        # test GNN+RL
        obs = env.reset(demand=demand,degree=degree,provide=True)
        done = False
        steps = 0
        while not done:
            action, _ = policy.predict(obs)
            obs, _, done, _ = env.step(action)
            steps += 1
        state_n = obs2adj(obs)
        print("final state (steps {})".format(steps))
        print(state_n)
        score_n = compute_reward(state_n, node_num, demand, degree, 50)
        scores_nn.append(score_n)

        print("[iter {0}][egotree:{1}][permatch:{2}][nn: {3}]".format(i_ter,score_e,score_m,score_n))
        """

        # test h w/o NN
        obs = env.reset(demand=demand,degree=degree,provide=True)
        done = False
        steps = 0
        while not done:
            graph, demand = obs
            action = opr.h_predict(graph,demand)
            obs, done = env.step(action)
            steps += 1
        graph, _ = obs
        state_h = np.array(nx.adjacency_matrix(graph).todense(),dtype=np.float32)
        #print("final state (steps {})".format(steps))
        print(state_h)
        score_h = compute_reward(state_h, node_num, demand, degree, 50)
        scores_h.append(score_h)
        steps_h.append(steps)

        print("[iter {0}][egotree:{1}][permatch:{2}][h: {3}]".format(i_ter,score_e,score_m,score_h))

    #print("Avg_scores: egotree{0} permatch{1} nn{2}".format(
    #    np.mean(scores_ego),np.mean(scores_match),np.mean(scores_nn)))
    print("Avg_scores: egotree{0} permatch{1} h{2}".format(
        np.mean(scores_ego),np.mean(scores_match),np.mean(scores_h)))
    print("Avg_steps_h: {}".format(np.mean(steps_h)))

def obs2adj(obs):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    adj = obs[:,:-1]
    adj[adj>0] = 1
    return adj

if __name__ == "__main__":
    main()