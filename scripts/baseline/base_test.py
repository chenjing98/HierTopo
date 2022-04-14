import numpy as np
import math
import pickle
from ego_tree import *
from permatch import permatch
from params import args

def cal_pathlength(state, node_num, demand, degree, degree_penalty):    
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
    penalty = args.penalty

    for i_ter in range(args.iters):
        with open(args.folder+str(node_num)+'demand_'+str(i_ter)+'.pk',"rb") as fp:
            demand = pickle.load(fp)
        with open(args.folder+str(node_num)+'degree_'+str(i_ter)+'.pk',"rb") as fp:
            degree = pickle.load(fp)          
        max_degree = max(degree)

        if args.baseline == 'egotree':
            test = ego_tree_unit(demand,node_num,degree,max_degree)
            test.create_tree()
            test.change_insert()
            state, count = test.estab()
        elif args.baseline == 'match':
            test = permatch(node_num)
            state = test.matching(demand, degree)
                
        score = cal_pathlength(state, node_num, demand, degree, 50)
        print(score)
