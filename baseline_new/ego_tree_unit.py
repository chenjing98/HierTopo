import numpy as np
import math

class ego_tree_unit:
    def __init__(self, demand, node_num, degree, max_degree):
        #params introduction
        #help record the insert position : help[e1*node_num+e2,0] record whether (e1,e2) is inserted
        #                                  help[e1*node_num+e2,1] is the edge to insert between e1 and e2
        #ego_tree is the union of all nodes' self tree (just for high degree nodes)
        #high and low record the high degree node and low degree nodes
        self.demand = demand
        self.node_num = node_num
        self.degree = degree
        self.help = np.zeros((node_num*node_num,2),dtype=int)
        for i in range(node_num*node_num):
            self.help[i,1] = -1
        self.max_degree = max_degree
        self.ego_tree = np.zeros((math.ceil(node_num/2),max_degree+1,node_num))
        for i in range(math.ceil(node_num/2)):
            for j in range(max_degree+1):
                for k in range(node_num):
                    self.ego_tree[i,j] = -np.inf
        self.high = np.argsort(self.degree)[math.ceil(node_num/2):node_num]
        self.low = np.argsort(self.degree)[0:math.ceil(node_num/2)]
        self.state = np.zeros((node_num,node_num))
        self.e_num = 0
        
    def create_tree(self):
        #create all high degree nodes' self tree
        for i in range(math.ceil(self.node_num/2)):
            index = self.high[i]
            #the first row of self tree is information
            self.ego_tree[i,0,0] = self.degree[index]
            self.ego_tree[i,0,1] = index
            p = self.demand[self.high[i],:]
            #Demand from high to low
            p_i = np.flipud(np.argsort(self.demand[index,:]))
            info = np.zeros((self.degree[index],2),dtype = int)
            child = 0
            #start put child node to self tree
            while(p[p_i[child]] > 0):
                if child < self.degree[index]:
                    #the child node connected to the root (itself) (called degree-tree)
                    self.ego_tree[i,child+1,0] = p_i[child]
                    #info is to record the degree-tree's total demand and number of nodes of the degree-tree
                    info[child,0] += 1
                    info[child,1] += p[p_i[child]]
                else:
                    #Nodes with a distance of at least 2 from the root (called 2-tree)
                    #choose which degree-tree to connect
                    choice = np.argmin(info[:,1])
                    self.ego_tree[i,choice+1,info[choice,0]] = p_i[child]
                    info[choice,0] += 1
                    info[choice,1] += p[p_i[child]]
                child += 1

    def change_insert(self):
        #find a low degree node to insert high degree edge and change the ego tree
        for i in range(math.ceil(self.node_num/2)):
            index = self.high[i]
            p_i = np.flipud(np.argsort(self.demand[index,:]))
            j = 0
            #make sure every edge of a high degree node can be inserted with different low degree node
            #and Try to ensure that all low points can be inserted
            if i%2 == 0:
                e_i = math.ceil(self.node_num/2) - 1
            else:
                e_i = 0
            while(self.demand[index,p_i[j]] > 0):
                #record insert
                if self.help[index*self.node_num + p_i[j],0] == 0 and (p_i[j] in self.high):
                    self.help[index*self.node_num + p_i[j],0] = 1
                    self.help[index*self.node_num + p_i[j],1] = self.low[e_i]
                    #this is an undirected graph
                    self.help[index + p_i[j] *self.node_num,0] = 1
                    self.help[index + p_i[j] *self.node_num,1] = self.low[e_i]
                    if i%2 == 0:
                        e_i -= 1
                    else:
                        e_i += 1
                j += 1
        #change ego tree according to help
        for i in range(math.ceil(self.node_num/2)):
            index = self.high[i]
            if self.ego_tree[i,0,0] > 0:
                for e in range(self.node_num):
                    if self.help[index * self.node_num + e,0] == 1 :
                        #find the position
                        insert = np.argwhere(self.ego_tree[i,1:,:] == e)
                        #Replace or remove this 'edge' node according to the rules
                        if self.demand[index,self.help[index * self.node_num + e,1]] == 0:
                            self.ego_tree[i,insert[0,0]+1,insert[0,1]] = self.help[index * self.node_num + e,1]
                        if self.demand[index,self.help[index * self.node_num + e,1]] > self.demand[index,e]:
                            self.ego_tree[i,insert[0,0]+1,insert[0,1]] = -np.inf
                        if self.demand[index,self.help[index * self.node_num + e,1]] < self.demand[index,e]:
                            self.ego_tree[i,insert[0,0]+1,insert[0,1]] = self.help[index * self.node_num + e,1]   

    def estab(self):
        #fill state according to ego tree
        for i in range(math.ceil(self.node_num/2)):
            index = self.high[i]
            tree = self.ego_tree[i,1:,:]
            for j in range(self.degree[index]):
                #if the degree-tree's root is removed so are its child node
                if tree[j,0] >= 0:
                    self.state[index,int(tree[j,0])] = 1
                    self.state[int(tree[j,0]),index] = 1
                    self.e_num += 1
                    #2-tree
                    for k in range(1,self.node_num):
                        if tree[j,k] >= 0:
                            #for a 2-tree we can get the node of a edge according to node farther from the root
                            n1 = int(tree[j,k])
                            z1 = tree[j,k]+1
                            z = math.log(z1,2)
                            c =2**(math.ceil(z)) - 1
                            o = math.ceil((j - c +1)/2)
                            n = int((c + 1)/2 -1 + o - 1)
                            #the node is existed
                            if tree[j,n] >= 0:
                                n2 = int(tree[j,n])
                                #because of the insert
                                if n1 != n2:
                                    self.state[n1,n2] = 1
                                    self.state[n2,n1] = 1
                                    self.e_num += 1
        return self.state,self.e_num
        
