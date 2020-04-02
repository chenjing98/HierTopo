import numpy as np
import math
from ego_tree_unit import *
from permatch import *
def compute_reward(state, node_num, demand, degree,degree_penalty):    
    D = state.copy()           
    for i in range(node_num):
        for j in range(node_num):
            if (D[i][j] == 0) & (i != j):
                D[i][j] = 999
    for k in range(node_num):
        for i in range(node_num):
            for j in range(node_num):
                if(D[i][j]>D[i][k]+D[k][j]):
                    D[i][j]=D[i][k]+D[k][j]
    score = 0
    for i in range(node_num-1):
        for j in range(i+1,node_num):
            if(D[i,j]) > node_num:
                score += demand[i,j]*node_num
            else:
                score += demand[i,j]*D[i,j] 

    for i in range(node_num):
        if np.sum(state[i,:]) > degree[i]:
            score = score + degree_penalty            
    return score 


if __name__ == "__main__":
    for i in range(6,30,2):
        result = np.empty((0,2))
        result_a = 0
        result_b = 0
        for i_ter in range(1000):
            node_num = i
            degree = np.random.randint(1, high = i, size =(i))
            max_degree = max(degree)
            demand = np.random.randint(0, high=50, size=(i,i))
            demand = np.triu(demand)
            demand += demand.T
            demand -= np.diag(demand.diagonal())
            test_a = ego_tree_unit(demand,node_num,degree,max_degree)
            test_a.create_tree()
            test_a.change_insert()
            state_a, count = test_a.estab()
            test_b = permatch(demand,node_num,degree)
            state_b = test_b.matching(count)
            score_a = compute_reward(state_a, node_num, demand, degree, 50)
            score_b = compute_reward(state_b, node_num, demand, degree, 50)
            result_a += score_a
            result_b += score_b
        result_a = result_a/1000
        result_b = result_b/1000
        r = np.zeros((1,2))
        r[0,0] = result_a
        r[0,1] = result_b
        result = np.append(result,r,axis =0)
        print('OK!')

