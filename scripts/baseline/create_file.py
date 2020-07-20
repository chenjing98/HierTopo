import pickle
import numpy as np

class create_file:
    def __init__(self, node_num):
        self.node_num = node_num
    
    def create(self, half_maxdemand, max_degree, folder="./data/"):
        testset = []
        for _ in range(10000):
            demand = np.random.randint(0, high=half_maxdemand, size=(self.node_num,self.node_num))
            demand = np.triu(demand)
            demand += demand.T
            demand -= np.diag(demand.diagonal())
            degree = np.random.randint(1, high=max_degree, size=(self.node_num))
            testset.append({'demand': demand, 'allowed_degree': degree})
        file_name = folder+'test'+str(self.node_num)+'_'+str(2*half_maxdemand)+'_'+str(max_degree)+'.pk'
        with open(file_name,"wb") as fp:
            pickle.dump(testset,fp)