import numpy as np
import math

class permatch(object):
    def __init__(self, node_num):
        self.node_num = node_num
        self.inf = 1000 # unnecessary
        #self.degree = degree
        #self.demand = demand
        #self.state = np.zeros((self.node_num,self.node_num))  
    
    def matching(self, demand, degree):
        demand_vec = []
        allowed_degree = degree
        #this is an undirected graph
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                demand_vec.append(demand[i,j]+demand[j,i])
        state = np.zeros((self.node_num,self.node_num))  

        for _ in range(int(self.node_num*(self.node_num-1)/2)):
            #choose an edge
            e = demand_vec.index(max(demand_vec))
            #edge.append(e)
            n = self.edge_to_node(e)
            n1 = n[0]
            n2 = n[1]
            if allowed_degree[n1] > 0 and allowed_degree[n2] > 0:
                state[n1,n2] = 1
                state[n2,n1] = 1
                allowed_degree[n1] -= 1
                allowed_degree[n2] -= 1
            #make this edge not be used again
            demand_vec[e] = -self.inf
        return state

    #the order of edge ---> the order of two nodes
    def edge_to_node(self,e):
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                if ((i*(2*self.node_num-1-i)/2-1+j-i)== e):
                    return [i,j]