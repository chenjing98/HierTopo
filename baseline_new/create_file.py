import pickle
import numpy as np
from params import args

class create_file:
    def __init__(self, node_num):
        self.node_num = node_num
    
    def create(self):
        for i in range(10000):
            demand = np.random.randint(0, high=args.maxdemand, size=(self.node_num,self.node_num))
            demand = np.triu(demand)
            demand += demand.T
            demand -= np.diag(demand.diagonal())
            degree = np.random.randint(1, high = self.node_num, size =(self.node_num))
            with open(args.folder+str(self.node_num)+'demand_'+str(i)+'.pk',"wb") as fp:
                pickle.dump(demand,fp)
            with open(args.folder+str(self.node_num)+'degree_'+str(i)+'.pk',"wb") as fp:
                pickle.dump(degree,fp)