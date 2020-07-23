import copy
import itertools
import numpy as np
import networkx as nx
import pickle as pk

from baseline.ego_tree import ego_tree_unit
from baseline.permatch import permatch
from whatisoptimal import optimal
from param_search.OptSearch import TopoOperator, dict2dict, dict2nxdict
from param_search.plotv import TopoSimulator

schemes = ["optimal", "weighted-matching"] # options: "optimal", "weighted-matching", "egotree", "param-search", "rl"
data_source = "nsfnet" # options: "random", "nsfnet", "geant2"
n_steps = 1

# parameters for "search"
alpha_v = 1.2
alpha_i = 0.1

# parameters for supervised learning & reinforcement learning
Max_degree = 4
dims = [3, 64, 1]
model_name_sl = "../saved_model/model"
model_name_rl = "../saved_model/gnn_ppo4topo1"

if "rl" in schemes:
    from stable_baselines import PPO2
if "sl" in schemes:
    import tensorflow as tf
    from SL.SLmodel import supervisedModel
if "rl" in schemes or "sl" in schemes:
    from RL.topoenv_backup import TopoEnv

if data_source == "random":
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

    # initialize models
    if "weighted-matching" in schemes:
        permatch_model = permatch(node_num)
    if "optimal" in schemes:
        opt = optimal()
    if "param-search" in schemes:
        opr = TopoOperator(node_num)
        opr.reset(alpha_v,alpha_i)
        sim = TopoSimulator(n_node=node_num)
    if "rl" in schemes:
        policy = PPO2.load(model_name_rl)
        env = TopoEnv(node_num)
    #if "sl" in schemes:
    #    sess = tf.Session()
    #    model = supervisedModel(sess, node_num, 4, [3,64,1])
    #    ckpt = tf.train.get_checkpoint_state(model_name_sl)
    #    saver = tf.train.Saver()
    #    saver.restore(sess,ckpt.model_checkpoint_path)
    #    env = TopoEnv(node_num)

    # load dataset
    with open(file_demand_degree, 'rb') as f1:
        dataset = pk.load(f1)
    with open(file_topo, 'rb') as f2:
        dataset_topo = pk.load(f2)

    # start testing
    for i_iter in range(n_iters):
        if data_source == "random":
            demand = dataset[i_iter]['demand']
            degree = dataset[i_iter]['allowed_degree']
            topo = dataset_topo[i_iter]
        else:
            demand = dataset[i_iter]
            degree = Max_degree * np.ones((node_num,), dtype=np.float32)
            topo = dataset_topo

        print("************** iter {} **************".format(i_iter))

        if "egotree" in schemes:
            int_degree = degree.astype(int) 
            max_degree = int(max(degree))
            test_e = ego_tree_unit(demand,node_num,int_degree,max_degree)
            test_e.create_tree()
            test_e.change_insert()
            state_e, _ = test_e.estab()
            cost_e = compute_reward(state_e, node_num, demand, degree)
            costs_ego.append(cost_e)
            print("egotree: {}".format(cost_e))

        if "optimal" in schemes:
            _, opt_dict = opt.multistep_DFS(node_num,topo,demand,degree,n_steps)
            opt_graph = nx.from_dict_of_dicts(opt_dict)
            state_o = np.array(nx.adjacency_matrix(opt_graph).todense(), np.float32)
            cost_o = compute_reward(state_o, node_num, demand, degree)
            costs_opt.append(cost_o)
            print("optimal: {}".format(cost_o))
            #v_optimal = opt.consturct_v(best_action,neigh)

        if "weighted-matching" in schemes:
            origin_graph = nx.from_dict_of_dicts(topo)
            permatch_new_graph = permatch_model.n_steps_matching(
                    demand,origin_graph,degree,n_steps)
            state_m = np.array(nx.adjacency_matrix(permatch_new_graph).todense(), np.float32)
            cost_m = compute_reward(state_m, node_num, demand, degree)
            costs_match.append(cost_m)
            print("weighted_matching: {}".format(cost_m))

        if "param-search" in schemes:
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

        if "rl" in schemes:
            obs = env.reset(demand=demand,degree=degree,provide=True)
            for _ in range(n_steps):
                action, _ = policy.predict(obs)
                obs, _, _, _ = env.step(action)
            state_rl = obs2adj(obs,node_num)
            cost_rl = compute_reward(state_rl, node_num, demand, degree)
            costs_rl.append(cost_rl)
            print("RL: {}".format(cost_rl))
        
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

        ## test per-match
        #state_m = permatch_model.matching(demand,degree)
        #cost_m = compute_reward(state_m, node_num, demand, degree, 100)
        #costs_match.append(cost_m)

        ## optimal (1 step)
        #origin_graph = nx.from_dict_of_dicts(topo)
        #best_action,neigh,opt_graph = opt.compute_optimal(node_num,origin_graph,demand,degree)
        #state_o1 = np.array(nx.adjacency_matrix(opt_graph).todense(), np.float32)
        #cost_o1 = compute_reward(state_o1, node_num, demand, degree, 100)
        #opt_dict = opt.multistep_compute_optimal(node_num,topo,demand,degree,n_steps)

        ## optimal overall (not limited by n_steps)
        #cost, _ = opt.optimal_topology(node_num, demand, degree)
        #cost_o_best = cost/(sum(sum(demand)))   
        #cost_opt_best.append(cost_o_best)

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

        
    print("============= Avg_costs ({} step(s)) =============".format(n_steps))
    if "optimal" in schemes:
        print("optimal: {}".format(np.mean(costs_opt)))
    if "weighted-matching" in schemes:
        print("weighted-matching: {}".format(np.mean(costs_match)))
    if "egotree" in schemes:
        print("egotree: {}".format(np.mean(costs_ego)))
    if "param-search" in schemes:
        print("search: {}".format(np.mean(costs_search)))
    if "rl" in schemes:
        print("RL: {}".format(np.mean(costs_rl)))
    
    

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