import pickle as pk
import numpy as np
import math

# Set parameters
data_count = 10000
dataset = []
num_nodes = 15
max_degree = 4
distribution = 'logistic'
file_name = '../../data/traindata/{0}_{1}_{2}_{3}.pk3'.format(data_count, num_nodes, max_degree, distribution)

def poisson_demand():
    allowed_degree = np.ones(num_nodes) * max_degree
    for _ in range(data_count):
        demand = np.random.poisson(lam=3.0, size=(num_nodes, num_nodes))
        np.fill_diagonal(demand, 0.0)  
        dataset.append({'demand': demand, 'allowed_degree': allowed_degree})
        
    with open(file_name, 'wb') as f:
        pk.dump(dataset, f)

def logistic_demand(density=1.0):
    mu = 2.63054
    gamma = 0.064096
    n_ones = math.floor(num_nodes*(num_nodes-1)*density)
    origin_arr = np.zeros((num_nodes*(num_nodes-1),),np.float32)
    origin_arr[:n_ones] = 1.0
    for i in range(data_count):
        demand_log = np.random.logistic(loc=mu, scale=gamma, size=(num_nodes, num_nodes))
        demand = np.power(10.0, demand_log)
        np.fill_diagonal(demand, 0.0)
        if density < 1.0:
            mask = create_mask(origin_arr)
            demand = np.multiply(mask, demand)
        dataset.append(demand)
        print("[no {0}] demand:\n {1}".format(i, demand))
    with open(file_name, 'wb') as f:
        pk.dump(dataset, f)

def create_mask(origin_arr):
    mat = np.reshape(np.random.shuffle(origin_arr), (num_nodes-1, num_nodes))
    diag = np.ones((num_nodes-1,1), np.float32)
    mat = np.reshape(np.hstack((diag, mat)),(num_nodes**2-1))
    mat = np.append(mat, 1.0)
    mat = np.reshape(mat, (num_nodes, num_nodes))
    return mat

if distribution == 'poisson':
    poisson_demand()
elif distribution == 'logistic':
    logistic_demand()

