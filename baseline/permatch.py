import numpy as np
import math
class permatch:
    def __init__(self, demand, node_num, degree):
        self.node_num = node_num
        self.degree = degree
        self.demand = demand
        self.state = np.zeros((self.node_num,self.node_num))  
    
    def matching(self,count):
        demand = []
        allowed_degree = self.degree
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                demand.append(self.demand[i,j])
        edge = []  
        c = 0
        error = 1
        error_num = 0
        while c < count:
            e = demand.index(max(demand))
            edge.append(e)
            n = self.edge_to_node(e)
            n1 = n[0]
            n2 = n[1]
            if allowed_degree[n1] > 0 and allowed_degree[n2] > 0:
                self.state[n1,n2] = 1
                self.state[n2,n1] = 1
                allowed_degree[n1] -= 1
                allowed_degree[n2] -= 1
                c += 1
                error = 1
            else: 
                if error == 0:
                    error_num += 1
                    if error_num > self.node_num:
                        return self.state
                error = 0
            demand[e] = -1000 
        return self.state

    def edge_to_node(self,e):
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                if ((i*(2*self.node_num-1-i)/2-1+j-i)== e):
                    return [i,j]