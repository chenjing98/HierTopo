import numpy as np
import math
import pickle
from ego_tree_unit import *
from permatch import *
from create_file import *
from params import args

def compute_reward(state, node_num, demand, degree,degree_penalty):    
    D = state.copy()       

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


if __name__ == "__main__":
    node_num = args.node
    
    f = create_file(node_num)
    f.create()
    
    penalty = args.penalty
    if args.baseline == 'egotree':
        for i_ter in range(args.iters):
            with open(args.folder+str(node_num)+'demand_'+str(i_ter)+'.pk',"rb") as fp:
                demand = pickle.load(fp)
            with open(args.folder+str(node_num)+'degree_'+str(i_ter)+'.pk',"rb") as fp:
                degree = pickle.load(fp)          
            max_degree = max(degree)
            test_a = ego_tree_unit(demand,node_num,degree,max_degree)
            test_a.create_tree()
            test_a.change_insert()
            state_a, count = test_a.estab()
            score_a = compute_reward(state_a, node_num, demand, degree, 50)
            print(score_a)
    
    if args.baseline == 'match':
        for i_ter in range(args.iters):
            with open(args.folder+str(node_num)+'demand_'+str(i_ter)+'.pk',"rb") as fp:
                demand = pickle.load(fp)
            with open(args.folder+str(node_num)+'degree_'+str(i_ter)+'.pk',"rb") as fp:
                degree = pickle.load(fp)          
            max_degree = max(degree)
            test_b = permatch(demand,node_num,degree)
            state_b = test_b.matching(args.count)
            score_b = compute_reward(state_b, node_num, demand, degree, 50)
            print(score_b)
        

