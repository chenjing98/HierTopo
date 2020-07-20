import pickle as pk
import numpy as np

data_count = 10000
dataset = []
num_nodes = 8
allowed_degree = np.ones(num_nodes) * 4
file_name = '10000_8_4_test.pk3'

for i in range(data_count):
    demand = np.random.poisson(lam=3.0, size=(num_nodes, num_nodes))
    np.fill_diagonal(demand, 0.0)  
    dataset.append({'demand': demand, 'allowed_degree': allowed_degree})
    
with open(file_name, 'wb') as f:
    pk.dump(dataset, f)
