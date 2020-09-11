import numpy as np
import networkx as nx

import pygad

import pickle as pk
import itertools
import random
import multiprocessing

"""
Given the following function:
    y = f(alpha, x)
    where y gets its minimum
What are the best values for the weights (alpha)?
"""
k = 6
n_iters = 3
n_steps = 4
#degree_lim = 8
parallelism = 10
data_source = "random"

if data_source == "random":
    node_num = 8
    degree_lim = 4
    n_testings = 1000
    file_demand_degree = '../../data/10000_8_4_test.pk3'
    file_topo = "../../data/10000_8_4_topo_test.pk3"
elif data_source == "nsfnet":
    node_num = 14
    degree_lim = 4
    n_testings = 100
    file_demand_degree = '../../data/nsfnet/demand_100.pkl'
    file_topo = '../../data/nsfnet/topology.pkl'
elif data_source == "geant2":
    node_num = 24
    degree_lim = 8
    n_testings = 100
    file_demand_degree = '../../data/geant2/demand_100.pkl'
    file_topo = '../../data/geant2/topology.pkl'
else:
    print("data_source {} unrecognized.".format(data_source))
    exit(1)

print("Settings:\ndata source = {0}\nn_steps     = {1}\nn_iters     = {2}\nparallelism = {3}".format(data_source,n_steps,n_iters,parallelism))

desired_output = 0.99 # Function output.

with open(file_demand_degree, 'rb') as f1:
    dataset = pk.load(f1)
with open(file_topo, 'rb') as f2:
    dataset_topo = pk.load(f2)

def apply_policy(demand, topo, alpha):
    """
    :param demand: (np.array) N x N
    :param topo: (nx.dict_of_dicts)
    :param alpha: (np.array) N
    :return: metric: (np.float32) average shortest path length
    """
    
    path_length = 0
    # normalize demand
    x = demand/np.max(demand)*2

    graph = nx.from_dict_of_dicts(topo)
    adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
    degree = np.sum(adj, axis=-1)

    for s in range(n_steps):
        for i in range(n_iters):
            exp_x = expand_orders_mat(x)
            weighing_self = np.matmul(exp_x, alpha[2*i*k:(2*i+1)*k])
            weighing_neigh = np.matmul(exp_x, alpha[(2*i+1)*k:(2*i+2)*k])
            neighbor_aggr = np.matmul(weighing_neigh, adj)
            g = weighing_self + neighbor_aggr
            x = g/np.max(g)*2 # N x N
        
        v = np.sum(x, axis=0)
        dif = cal_diff(v)
        ind_x, ind_y = np.where(dif==np.max(dif))
        if len(ind_x) < 1:
            continue
        elif len(ind_x) > 1:
            s = random.randint(0, len(ind_x)-1)
            add_ind = (ind_x[s], ind_y[s])
        else:
            add_ind = (ind_x[0], ind_y[0])

        if adj[add_ind] == 1:
            path_length += 1.0
        if degree[add_ind[0]] >= degree_lim:
            path_length += 0.5
        if degree[add_ind[0]] >= degree_lim:
            path_length += 0.5
        else:
            graph.add_edge(add_ind[0], add_ind[1])
            adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
            degree = np.sum(adj, axis=-1)
            x = demand/np.max(demand)*2
    
    path_length += cal_pathlength(demand, graph)
    return path_length


def expand_orders_mat(feature):
    """
    :param feature: (np.array) N x N
    :return exp_feature: (np.array) N x N x k
    """
    N = feature.shape[0]
    exp_feature = np.zeros((N,N,k), np.float32)
    for i in range(k):
        exp_feature[:, :, i] = np.power(feature, i)
    return exp_feature

def cal_diff(v):
    N = v.shape[0]
    dif = np.repeat(np.expand_dims(v,0),N,0) - np.repeat(np.expand_dims(v,-1),N,-1)
    dif = np.abs(dif)
    return dif
    

def cal_pathlength(demand, graph):
    n_nodes = demand.shape[0]
    score = 0
    for s, d in itertools.product(range(n_nodes), range(n_nodes)):
        try:
            cur_path_length = float(nx.shortest_path_length(graph,source=s,target=d))
        except nx.exception.NetworkXNoPath:
            cur_path_length = float(n_nodes)

        score += cur_path_length * demand[s,d]
    score /= (sum(sum(demand)))
    return score

def test(solution, test_size):
    metrics = []
    q = multiprocessing.Queue()
    lock = multiprocessing.Lock()
    processes = []
    for i in range(parallelism):
        p = multiprocessing.Process(target=test_run,args=(lock, q, solution, test_size, i,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    for _ in range(parallelism):
        metric = q.get()
        metrics += metric

    output = np.mean(metrics)
    return output

def test_run(lock, queue, solution, test_size, process_no):
    m_tmp = []
    for i in range(test_size):
        if i % parallelism != process_no:
            continue
        if data_source == 'random':
            m = apply_policy_robust(dataset[i]['demand'], dataset_topo[i], solution)
        else:
            m = apply_policy_robust(dataset[i], dataset_topo, solution)
        m_tmp.append(m)
    #print("finished calculating. {}".format(process_no))
    lock.acquire()
    queue.put(m_tmp)
    lock.release()

def apply_policy_robust(demand, topo, alpha):
    """
    :param demand: (np.array) N x N
    :param topo: (nx.dict_of_dicts)
    :param alpha: (np.array) N
    :return: metric: (np.float32) average shortest path length
    """
    
    path_length = 0
    # normalize demand
    x = demand/np.max(demand)*2

    graph = nx.from_dict_of_dicts(topo)
    adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
    degree = np.sum(adj, axis=-1)
    n_nodes = adj.shape[0]

    for _ in range(n_steps):
        for _ in range(n_iters):
            exp_x = expand_orders_mat(x)
            weighing_self = np.matmul(exp_x, alpha[:k])
            weighing_neigh = np.matmul(exp_x, alpha[k:2*k])
            neighbor_aggr = np.matmul(weighing_neigh, adj)
            g = weighing_self + neighbor_aggr
            x = g/np.max(g)*2 # N x N
        
        v = np.sum(x, axis=0)
        dif = cal_diff(v)
        degree_full = np.where(degree>=degree_lim, 1.0, 0.0)
        degree_mask = np.repeat(np.expand_dims(degree_full,0),n_nodes,0) + np.repeat(np.expand_dims(degree_full,-1),n_nodes,-1)
        mask = adj + np.identity(n_nodes, np.float32) + degree_mask
        masked_dif = (mask == 0) * dif
        ind_x, ind_y = np.where(masked_dif==np.max(masked_dif))
        if len(ind_x) < 1:
            continue
        elif len(ind_x) > 1:
            s = random.randint(0, len(ind_x)-1)
            add_ind = (ind_x[s], ind_y[s])
        else:
            add_ind = (ind_x[0], ind_y[0])

        if (adj[add_ind] != 1) and (degree[add_ind[0]] < degree_lim) and (degree[add_ind[0]] < degree_lim):
            graph.add_edge(add_ind[0], add_ind[1])
            adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
            degree = np.sum(adj, axis=-1)
            x = demand/np.max(demand)*2
    
    path_length = cal_pathlength(demand, graph)
    return path_length

def test_robust(solution, test_size):
    metrics = []
    for i in range(test_size):
        if data_source == 'random':
            m = apply_policy_robust(dataset[i]['demand'], dataset_topo[i], solution)
        else:
            m = apply_policy_robust(dataset[i], dataset_topo, solution)
        metrics.append(m)
        #print("[No. {0}] {1}".format(i,m))
    output = np.mean(metrics)
    return output

def fitness_func(solution, solution_idx):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    output = test(solution, n_testings)
    fitness = 1.0 / np.abs(output - desired_output)
    #print("solution {0} metric {1}".format(solution, output))
    return fitness

fitness_function = fitness_func

num_generations = 200 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.

# To prepare the initial population, there are 2 ways:
# 1) Prepare it yourself and pass it to the initial_population parameter. This way is useful when the user wants to start the genetic algorithm with a custom initial population.
# 2) Assign valid integer values to the sol_per_pop and num_genes parameters. If the initial_population parameter exists, then the sol_per_pop and num_genes parameters are useless.
sol_per_pop = 50 # Number of solutions in the population.
num_genes = 2 * k * n_iters

init_range_low = -2
init_range_high = 5

parent_selection_type = "sss" # Type of parent selection.
keep_parents = 7 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

crossover_type = "single_point" # Type of the crossover operator.

# Parameters of the mutation operation.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists or when mutation_type is None.

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    print("Change     = {change}".format(change=ga_instance.best_solution()[1] - last_fitness))
    last_fitness = ga_instance.best_solution()[1]

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       callback_generation=callback_generation)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
#ga_instance.plot_result()

print("Settings:\ndata source = {0}\nn_steps     = {1}\nn_iters     = {2}\nn_orders    = {3}\nparallelism = {4}".format(data_source,n_steps,n_iters,k,parallelism))

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

prediction = test_robust(solution, n_testings)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

# Saving the GA instance.
filename = 'genetic_diffparam_{0}_{1}step_{2}order'.format(data_source,n_steps,k) # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
#loaded_ga_instance = pygad.load(filename=filename)
#loaded_ga_instance.plot_result()

