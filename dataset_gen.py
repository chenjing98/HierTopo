import pickle as pk
import numpy as np

data_count = 10000
dataset = []
num_nodes = 3

for i in range(data_count):
    demand = np.random.poisson(lam=3.0, size=(num_nodes, num_nodes))
    allowed_degree = np.ones(num_nodes) * 3
    dataset.append({'demand': demand, 'allowed_degree': allowed_degree})
    
with open('10M_8_3.0_const3.pk3', 'wb') as f:
    pk.dump(dataset, f)
