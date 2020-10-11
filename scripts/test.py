import copy
import itertools
import numpy as np
import networkx as nx
import pickle as pk
from timeit import default_timer as timer

from baseline.ego_tree import ego_tree_unit
from baseline.permatch import permatch
from whatisoptimal import optimal
from param_search.OptSearch import TopoOperator, dict2dict, dict2nxdict
from param_search.plotv import TopoSimulator
from baseline.bmatching import bMatching
from multiprocessing import Pool

methods = ["oblivious-opt"] # options: "optimal", "greedy", "egotree", "param-search", "rl", "bmatch", "optimal-mp", "oblivious"
data_source = "scratch" # options: "random8", "nsfnet", "geant2", "scratch"
scheme = "complete" # options: "complete", "bysteps"
Max_degree = 4
n_steps = 2
n_nodes = 50

# parameters for "search"
alpha_v = 1.2
alpha_i = 0.1

# parameters for supervised learning & reinforcement learning
dims = [3, 64, 1]
model_name_sl = "../saved_model/model"
model_name_rl = "../saved_model/gnn_ppo4topo1"

if "rl" in methods:
    from stable_baselines import PPO2
if "sl" in methods:
    import tensorflow as tf
    from SL.SLmodel import supervisedModel
if "rl" in methods or "sl" in methods:
    from RL.topoenv_backup import TopoEnv

if data_source == "random8":
    node_num = 8
    n_iters = 1000
    file_demand_degree = '../data/10000_8_4_test.pk3'
    file_topo = "../data/10000_8_4_topo_test.pk3"
elif data_source == "nsfnet":
    node_num = 14
    n_iters = 100
    file_demand_degree = '../data/nsfnet/demand_100.pkl'
    file_topo = '../data/nsfnet/topology.pkl'
elif data_source == "geant2":
    node_num = 24
    n_iters = 100
    file_demand_degree = '../data/geant2/demand_100.pkl'
    file_topo = '../data/geant2/topology.pkl'
elif data_source == "germany":
    node_num = 50
    n_iters = 100
    file_demand_degree = '../data/germany/demand_100.pkl'
    file_topo = '../data/germany/topology.pkl'
elif data_source == "scratch":
    node_num = n_nodes
    n_iters = 1000
    file_demand = '../data/2000_{0}_{1}_logistic.pk3'.format(n_nodes, Max_degree)
else:
    print("data_source {} unrecognized.".format(data_source))
    exit(1)

def compute_reward(state, num_node, demand, degree):    
    D = copy.deepcopy(state)
    graph = nx.from_numpy_matrix(D)
    cost = 0
    for s, d in itertools.product(range(num_node), range(num_node)):
        try:
            path_length = float(nx.shortest_path_length(graph,source=s,target=d))
        except nx.exception.NetworkXNoPath:
            path_length = float(num_node)

        cost += path_length * demand[s,d]   

    cost /= (sum(sum(demand)))
    return cost

def main():
    costs_opt = []
    costs_ego = []
    costs_match = []
    costs_search = []
    costs_rl = []
    costs_b = []
    costs_ob = []

    t_begin = timer()
    # initialize models
    if "greedy" in methods:
        permatch_model = permatch(node_num)
    if "optimal" in methods or "optimal-mp" in methods or "oblivious-opt" in methods:
        opt = optimal()
    if "param-search" in methods:
        opr = TopoOperator(node_num)
        opr.reset(alpha_v,alpha_i)
        sim = TopoSimulator(n_node=node_num)
    if "rl" in methods:
        policy = PPO2.load(model_name_rl)
        env = TopoEnv(node_num)
    if "bmatch" in methods:
        bmatch = bMatching(node_num, Max_degree)
    #if "sl" in methods:
    #    sess = tf.Session()
    #    model = supervisedModel(sess, node_num, 4, [3,64,1])
    #    ckpt = tf.train.get_checkpoint_state(model_name_sl)
    #    saver = tf.train.Saver()
    #    saver.restore(sess,ckpt.model_checkpoint_path)
    #    env = TopoEnv(node_num)

    # load dataset
    if data_source == "scratch":
        with open(file_demand, 'rb') as f:
            dataset = pk.load(f)
    else: 
        with open(file_demand_degree, 'rb') as f1:
            dataset = pk.load(f1)
        with open(file_topo, 'rb') as f2:
            dataset_topo = pk.load(f2)

    if "optimal-mp" in methods:
        params = []
        for i_iter in range(n_iters):
            param = {}
            param["demand"] = dataset[i_iter]
            param["degree"] = Max_degree
            param["n_nodes"] = n_nodes
            params.append(param)
        pool = Pool()
        costs = pool.map(opt.optimal_topology_run,params)
        costs_opt_mp = np.array(costs)
        pool.close()
        pool.join()
    elif "oblivious-opt" in methods:
        demand = np.zeros((n_nodes,n_nodes),np.float32)
        for i_iter in range(n_iters):
            damand += dataset[i_iter]
        demand /= n_iters
        degree = Max_degree * np.ones((node_num,), dtype=np.float32)
        cost_obl, graph_obl = opt.optimal_topology(node_num, demand, degree)
        cost = cost/(sum(sum(demand)))  
        print(graph_obl)
        print("N={0} cost={1}".format(node_num, cost_obl))
    else:
        # start testing
        for i_iter in range(n_iters):
            if data_source == "random8":
                demand = dataset[i_iter]['demand']
                degree = dataset[i_iter]['allowed_degree']
                topo = dataset_topo[i_iter]
            elif data_source == "scratch":
                demand = dataset[i_iter]
                degree = Max_degree * np.ones((node_num,), dtype=np.float32)
            else:
                demand = dataset[i_iter]
                degree = Max_degree * np.ones((node_num,), dtype=np.float32)
                topo = dataset_topo

            print("[iter {}]".format(i_iter))

            if "egotree" in methods:
                int_degree = degree.astype(int) 
                test_e = ego_tree_unit(demand,node_num,int_degree,Max_degree)
                test_e.create_tree()
                test_e.change_insert()
                state_e, _ = test_e.estab()
                print("adj: {}".format(state_e))
                cost_e = compute_reward(state_e, node_num, demand, degree)
                costs_ego.append(cost_e)
                print("egotree: {}".format(cost_e))
            
            if "bmatch" in methods:
                cost_b = bmatch.match(demand)
                #cost_b = compute_reward(state_b, node_num, demand, degree)
                costs_b.append(cost_b)
                print("bmatching: {}".format(cost_b))

            if "optimal" in methods:
                if scheme == "bysteps":
                    _, opt_dict = opt.multistep_DFS(node_num,topo,demand,degree,n_steps)
                    opt_graph = nx.from_dict_of_dicts(opt_dict)
                    state_o = np.array(nx.adjacency_matrix(opt_graph).todense(), np.float32)
                    cost_o = compute_reward(state_o, node_num, demand, degree)
                    costs_opt.append(cost_o)
                    print("optimal: {}".format(cost_o))
                    #v_optimal = opt.consturct_v(best_action,neigh)
                if scheme == "complete":
                    cost, _ = opt.optimal_topology(node_num, demand, degree)
                    cost_o = cost/(sum(sum(demand)))   
                    costs_opt.append(cost_o)
                    print("optimal: {}".format(cost_o))

            if "greedy" in methods:
                if scheme == "bysteps":
                    origin_graph = nx.from_dict_of_dicts(topo)
                    permatch_new_graph = permatch_model.n_steps_matching(
                            demand,origin_graph,degree,n_steps)
                    state_m = np.array(nx.adjacency_matrix(permatch_new_graph).todense(), np.float32)
                if scheme == "complete":
                    state_m = permatch_model.matching(demand,degree)
                cost_m = compute_reward(state_m, node_num, demand, degree)
                costs_match.append(cost_m)
                print("greedy: {}".format(cost_m))

            if "param-search" in methods:
                curr_graph = topo
                for i_step in range(n_steps):
                    v = opr.predict(curr_graph, demand)
                    if i_step < n_steps - 1:
                        curr_graph = sim.step_graph(node_num, v, 
                                                demand=demand, 
                                                topology=curr_graph,
                                                allowed_degree=degree)
                    else:
                        cost_s = sim.step(node_num, v, 
                                        demand=demand, 
                                        topology=curr_graph, 
                                        allowed_degree=degree)
                costs_search.append(cost_s)
                print("search: {}".format(cost_s))

            if "rl" in methods:
                obs = env.reset(demand=demand,degree=degree,provide=True)
                for _ in range(n_steps):
                    action, _ = policy.predict(obs)
                    obs, _, _, _ = env.step(action)
                state_rl = obs2adj(obs,node_num)
                cost_rl = compute_reward(state_rl, node_num, demand, degree)
                costs_rl.append(cost_rl)
                print("RL: {}".format(cost_rl))

            if "oblivious" in methods:
                state_ob = np.zeros((n_nodes,n_nodes))
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if (j-i)%n_nodes == 1 or (j-i)%n_nodes == 2 or (i-j)%n_nodes == 1 or (i-j)%n_nodes == 2:
                            state_ob[i,j] = 1
                cost_ob = compute_reward(state_ob, node_num, demand, degree)
                costs_ob.append(cost_ob)
                print("oblivious: {}".format(cost_ob))
        ## test supervised learning
        #obs = env.reset(demand=demand,degree=degree,provide=True)
        #demand_input, adj_input, deg_input = obs_process(obs, node_num)
        #potential = model.predict(demand_input,adj_input, deg_input)
        #action = np.squeeze(potential)
        #obs, _, _, _ = env.step(action)
        #state_sl = obs2adj(obs,node_num)
        #cost_sl = compute_reward(state_sl, node_num, demand, degree)
        #costs_sl.append(cost_sl)
        #print("SL: {}".format(cost_sl))

        ## optimal (1 step)
        #origin_graph = nx.from_dict_of_dicts(topo)
        #best_action,neigh,opt_graph = opt.compute_optimal(node_num,origin_graph,demand,degree)
        #state_o1 = np.array(nx.adjacency_matrix(opt_graph).todense(), np.float32)
        #cost_o1 = compute_reward(state_o1, node_num, demand, degree, 100)
        #opt_dict = opt.multistep_compute_optimal(node_num,topo,demand,degree,n_steps)

        ## parameter search (1 step)
        #v_weightedsum = opr.predict(topo,demand)
        #cost_s = sim.step(8,v_weightedsum,0,demand,topo,degree)

        ## test h w/o NN
        #obs = env.reset(demand=demand,degree=degree,provide=True)
        #done = False
        #steps = 0
        #while not done:
        #    graph, demand = obs
        #    action = opr.h_predict(graph,demand)
        #    obs, done = env.step(action)
        #    steps += 1
        #graph, _ = obs
        #state_h = np.array(nx.adjacency_matrix(graph).todense(),dtype=np.float32)
        #cost_h = compute_reward(state_h, node_num, demand, degree)

    t_end = timer()

    print("Setting:\ndata source = {0}\nn_nodes     = {1}".format(data_source, node_num))
    if scheme == "bysteps":
        print("======== Avg_costs & std ({} step(s)) ========".format(n_steps))
    elif scheme == "complete":
        print("========== Avg_costs & std (compl) ===========")
    
    if "optimal" in methods:
        print("optimal : {0}  std : {1}".format(np.mean(costs_opt),np.std(costs_opt)))
    if "optimal-mp" in methods:
        print("optimal : {0}  std : {1}".format(np.mean(costs_opt_mp),np.std(costs_opt_mp)))
    if "greedy" in methods:
        print("greedy  : {0}  std : {1}".format(np.mean(costs_match),np.std(costs_match)))
    if "egotree" in methods:
        print("egotree : {0}  std : {1}".format(np.mean(costs_ego),np.std(costs_ego)))
    if "bmatch" in methods:
        print("b-match : {0}  std : {1}".format(np.mean(costs_b),np.std(costs_b)))
    if "param-search" in methods:
        print("search  : {0}  std : {1}".format(np.mean(costs_search),np.std(costs_search)))
    if "rl" in methods:
        print("RL      : {0}  std : {1}".format(np.mean(costs_rl),np.std(costs_rl)))
    if "oblivious" in methods:
        print("oblivious:{0}  std : {1}".format(np.mean(costs_ob),np.std(costs_ob)))
    print("testing time : {} s".format(t_end-t_begin))
    
    

def obs2adj(obs,node_num):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    obs = np.reshape(obs,((2*node_num+1),node_num))
    adj = obs[node_num:-1,:]
    #adj[adj>0] = 1
    return adj

def obs_process(obs,node_num):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    obs = np.reshape(obs,((2*node_num+1),node_num))
    demand = obs[:node_num,:]
    adj = obs[node_num:-1,:]
    deg = obs[-1, :]
    #adj[adj>0] = 1
    return demand[np.newaxis,:], adj[np.newaxis,:], deg[np.newaxis,:]

if __name__ == "__main__":
    main()