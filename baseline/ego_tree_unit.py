import numpy as np
import math
class ego_tree_unit:
    def __init__(self, demand, node_num, degree, max_degree):
        self.demand = demand
        self.node_num = node_num
        self.degree = degree
        self.help = np.zeros((node_num*node_num,2),dtype=int)
        for i in range(node_num*node_num):
            self.help[i,1] = -10
        self.max_degree = max_degree
        self.ego_tree = np.zeros((math.ceil(node_num/2),max_degree+1,node_num))
        for i in range(math.ceil(node_num/2)):
            for j in range(max_degree+1):
                for k in range(node_num):
                    self.ego_tree[i,j] = -10
        self.high = np.argsort(self.degree)[math.ceil(node_num/2):node_num]
        self.low = np.argsort(self.degree)[0:math.ceil(node_num/2)]
        self.state = np.zeros((node_num,node_num))
        self.e_num = 0
        
    def create_tree(self):
        for i in range(math.ceil(self.node_num/2)):
            index = self.high[i]
            self.ego_tree[i,0,0] = self.degree[index]
            self.ego_tree[i,0,1] = index
            p = self.demand[self.high[i],:]
            p_i = np.flipud(np.argsort(self.demand[index,:]))
            info = np.zeros((self.degree[index],1),dtype = int)
            child = 0
            while(p[p_i[child]] > 0):
                if child < self.degree[index]:
                    self.ego_tree[i,child+1,0] = p_i[child]
                    info[child] += 1
                else:
                    choice = np.argmin(info)
                    self.ego_tree[i,choice+1,info[choice]] = p_i[child]
                    info[choice] += 1    
                child += 1

    def change_insert(self):
        
        for i in range(math.ceil(self.node_num/2)):
            index = self.high[i]
            p_i = np.flipud(np.argsort(self.demand[index,:]))
            j = 0
            if i%2 == 0:
                e_i = math.ceil(self.node_num/2) - 1
            else:
                e_i = 0
            while(self.demand[index,p_i[j]] > 0):
                if self.help[index*self.node_num + p_i[j],0] == 0 and (p_i[j] in self.high):
                    self.help[index*self.node_num + p_i[j],0] = 1
                    self.help[index*self.node_num + p_i[j],1] = self.low[e_i]
                    self.help[index + p_i[j] *self.node_num,0] = 1
                    self.help[index + p_i[j] *self.node_num,1] = self.low[e_i]
                    if i%2 == 0:
                        e_i -= 1
                    else:
                        e_i += 1
                j += 1
        for i in range(math.ceil(self.node_num/2)):
            index = self.high[i]
            if self.ego_tree[i,0,0] > 0:
                for edge in range(self.node_num):
                    if self.help[index * self.node_num + edge,0] == 1 :
                        insert = np.argwhere(self.ego_tree[i,1:,:] == edge)
                        if self.demand[index,self.help[index * self.node_num + edge,1]] == 0:
                            self.ego_tree[i,insert[0,0]+1,insert[0,1]] = self.help[index * self.node_num + edge,1]
                        if self.demand[index,self.help[index * self.node_num + edge,1]] > self.demand[index,edge]:
                            self.ego_tree[i,insert[0,0]+1,insert[0,1]] = -10
                        if self.demand[index,self.help[index * self.node_num + edge,1]] < self.demand[index,edge]:
                            self.ego_tree[i,insert[0,0]+1,insert[0,1]] = self.help[index * self.node_num + edge,1]   

    def estab(self):
        for i in range(math.ceil(self.node_num/2)):
            index = self.high[i]
            tree = self.ego_tree[i,1:,:]
            for j in range(self.degree[index]):
                if tree[j,0] >= 0:
                    self.state[index,int(tree[j,0])] = 1
                    self.state[int(tree[j,0]),index] = 1
                    if (int(tree[j,0])==index):
                        print(self.ego_tree[i,:,:])
                    self.e_num += 1
                    for k in range(1,self.node_num):
                        if tree[j,k] >= 0:
                            n1 = int(tree[j,k])
                            z1 = tree[j,k]+1
                            z = math.log(z1,2)
                            c =2**(math.ceil(z)) - 1
                            o = math.ceil((j - c +1)/2)
                            n = int((c + 1)/2 -1 + o - 1)
                            if tree[j,n] >= 0:
                                n2 = int(tree[j,n])
                                if n1 != n2:
                                    self.state[n1,n2] = 1
                                    self.state[n2,n1] = 1
                                    self.e_num += 1
        return self.state,self.e_num
        
