import copy
import itertools
import numpy as np
import networkx as nx
import pickle as pk
from timeit import default_timer as timer
import argparse

from baseline.ego_tree import ego_tree_unit
from baseline.permatch import permatch
from baseline.dijkstra_greedy import DijGreedyAlg
from whatisoptimal import optimal
from param_search.OptSearch import TopoOperator, dict2dict, dict2nxdict
from param_search.plotv import TopoSimulator
from baseline.bmatching import bMatching
from multiprocessing import Pool

# methods = ["bmatch", "greedy", "egotree", "dijgreedy"]
# methods = ["oblivious-opt"] # options: "optimal", "greedy", "egotree", "param-search", "rl", "bmatch", "optimal-mp", "oblivious"
data_source = "scratch"  # options: "random8", "nsfnet", "geant2", "germany", "scratch"
# scheme = "complete"  # options: "complete", "bysteps"
Max_degree = 4
n_steps = 20
n_node = 50

# parameters for "search"
alpha_v = 1.2
alpha_i = 0.1

# parameters for supervised learning & reinforcement learning
dims = [3, 64, 1]
model_name_sl = "../saved_model/model"
model_name_rl = "../saved_model/gnn_ppo4topo1"


def cal_pathlength(state, num_node, demand, degree):
    D = copy.deepcopy(state)
    graph = nx.from_numpy_matrix(D)
    cost = 0
    for s, d in itertools.product(range(num_node), range(num_node)):
        try:
            path_length = float(
                nx.shortest_path_length(graph, source=s, target=d))
        except nx.exception.NetworkXNoPath:
            path_length = float(num_node)

        cost += path_length * demand[s, d]

    cost /= (sum(sum(demand)))
    return cost


def check_connectivity(paths, i, j):
    if not i in paths:
        return False
    if not j in paths[i]:
        return False
    return True


def cal_change(adj, adj_prev, n_node):
    link_change = 0
    route_port_change = 0

    for i in range(n_node - 1):
        for j in range(i + 1, n_node):
            if not adj[i][j] == adj_prev[i][j]:
                link_change += 1

    G = nx.from_numpy_matrix(adj)
    G_prev = nx.from_numpy_matrix(adj_prev)
    paths = dict(nx.all_pairs_shortest_path(G))
    paths_prev = dict(nx.all_pairs_shortest_path(G_prev))
    for i in range(n_node):
        for j in range(n_node):
            if i == j:
                continue
            # print(i, j, paths[i][j], paths_prev[i][j])
            is_connected = check_connectivity(paths, i, j)
            is_connected_prev = check_connectivity(paths_prev, i, j)
            if (is_connected
                    and not is_connected_prev) or (is_connected_prev
                                                   and not is_connected):
                route_port_change += 1
                continue
            if not is_connected and not is_connected_prev:
                continue
            if not paths[i][j][1] == paths_prev[i][j][1]:
                route_port_change += 1
    return link_change, route_port_change


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-n",
                        "--n_node",
                        type=int,
                        help="Number of nodes in the network",
                        default=30)
    parser.add_argument("-s",
                        "--scheme",
                        type=str,
                        help="The topology adjustment scheme",
                        default="bysteps",
                        choices=["complete", "bysteps"])
    parser.add_argument("-m",
                        "--method",
                        type=str,
                        help="The method to be used",
                        default="greedy",
                        choices=[
                            "optimal", "greedy", "egotree", "param-search",
                            "rl", "bmatch", "optimal-mp", "oblivious", "dijgreedy"
                        ])
    args = parser.parse_args()
    
    scheme = args.scheme
    methods = []
    methods.append(args.method)
    
    if "rl" in methods:
        from stable_baselines import PPO2
    if "sl" in methods:
        import tensorflow as tf
        from SL.SLmodel import supervisedModel
    if "rl" in methods or "sl" in methods:
        from RL.topoenv_backup import TopoEnv

    if data_source == "random8":
        n_node = 8
        n_iters = 1000
        file_demand_degree = '../data/10000_8_4_test.pk3'
        file_topo = "../data/10000_8_4_topo_test.pk3"
    elif data_source == "nsfnet":
        n_node = 14
        n_iters = 100
        file_demand_degree = '../data/nsfnet/demand_100.pkl'
        file_topo = '../data/nsfnet/topology.pkl'
    elif data_source == "geant2":
        n_node = 24
        n_iters = 100
        file_demand_degree = '../data/geant2/demand_100.pkl'
        file_topo = '../data/geant2/topology.pkl'
    elif data_source == "germany":
        n_node = 50
        n_iters = 100
        file_demand_degree = '../data/germany/demand_100.pkl'
        file_topo = '../data/germany/topology.pkl'
    elif data_source == "scratch":
        n_node = args.n_node
        n_iters = 1000
        file_demand = '../data/2000_{0}_{1}_logistic.pk3'.format(
            n_node, Max_degree)
    else:
        print("data_source {} unrecognized.".format(data_source))
        exit(1)

    hops = {}
    steps = {}
    ports = {}
    adjs = {}
    for m in methods:
        hops[m] = []
        steps[m] = []
        ports[m] = []

    t_begin = timer()
    # initialize models
    if "greedy" in methods:
        permatch_model = permatch(n_node)
    if "dijgreedy" in methods:
        dijgreedy_model = DijGreedyAlg(n_node, Max_degree)
    if "optimal" in methods or "optimal-mp" in methods or "oblivious-opt" in methods:
        opt = optimal()
    if "param-search" in methods:
        opr = TopoOperator(n_node)
        opr.reset(alpha_v, alpha_i)
        sim = TopoSimulator(n_node=n_node)
    if "rl" in methods:
        policy = PPO2.load(model_name_rl)
        env = TopoEnv(n_node)
    if "bmatch" in methods:
        bmatch = bMatching(n_node, Max_degree)
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
            param["n_nodes"] = n_node
            params.append(param)
        pool = Pool()
        costs = pool.map(opt.optimal_topology_run, params)
        costs_opt_mp = np.array(costs)
        pool.close()
        pool.join()
    elif "oblivious-opt" in methods:
        demand = np.zeros((n_node, n_node), np.float32)
        for i_iter in range(n_iters):
            demand += dataset[i_iter]
        demand /= n_iters
        degree = Max_degree * np.ones((n_node, ), dtype=np.float32)
        cost_obl, graph_obl = opt.optimal_topology(n_node, demand, degree)
        cost = cost / (sum(sum(demand)))
        print(graph_obl)
        print("N={0} cost={1}".format(n_node, cost_obl))
    else:
        # start testing
        for i_iter in range(n_iters):
            if data_source == "random8":
                demand = dataset[i_iter]['demand']
                degree = dataset[i_iter]['allowed_degree']
                topo = dataset_topo[i_iter]
            elif data_source == "scratch":
                demand = dataset[i_iter]
                degree = Max_degree * np.ones((n_node, ), dtype=np.float32)
            else:
                demand = dataset[i_iter]
                degree = Max_degree * np.ones((n_node, ), dtype=np.float32)
                topo = dataset_topo

            print("[iter {}]".format(i_iter))

            if "egotree" in methods:
                int_degree = degree.astype(int)
                test_e = ego_tree_unit(demand, n_node, int_degree, Max_degree)
                test_e.create_tree()
                test_e.change_insert()
                state_e, _ = test_e.estab()
                # print("adj: {}".format(state_e))
                hop_e = cal_pathlength(state_e, n_node, demand, degree)
                if i_iter > 0:
                    step, port = cal_change(state_e, adjs["egotree"], n_node)
                    steps["egotree"].append(step)
                    ports["egotree"].append(port)
                adjs["egotree"] = state_e
                hops["egotree"].append(hop_e)
                print("egotree: {}".format(hop_e))

            if "bmatch" in methods:
                state_b = bmatch.match(demand)
                cost_b = cal_pathlength(state_b, n_node, demand, degree)
                hops["bmatch"].append(cost_b)
                if i_iter > 0:
                    step, port = cal_change(state_b, adjs["bmatch"], n_node)
                    steps["bmatch"].append(step)
                    ports["bmatch"].append(port)
                adjs["bmatch"] = state_b
                print("bmatching: {}".format(cost_b))

            if "optimal" in methods:
                if scheme == "bysteps":
                    _, opt_dict = opt.multistep_DFS(n_node, topo, demand,
                                                    degree, n_steps)
                    opt_graph = nx.from_dict_of_dicts(opt_dict)
                    state_o = np.array(
                        nx.adjacency_matrix(opt_graph).todense(), np.float32)
                    cost_o = cal_pathlength(state_o, n_node, demand, degree)
                    hops["optimal"].append(cost_o)
                    if i_iter > 0:
                        step, port = cal_change(state_o, adjs["optimal"],
                                                n_node)
                        steps["optimal"].append(step)
                        ports["optimal"].append(port)
                    adjs["optimal"] = state_o
                    print("optimal: {}".format(cost_o))
                    #v_optimal = opt.consturct_v(best_action,neigh)
                if scheme == "complete":
                    cost, _ = opt.optimal_topology(n_node, demand, degree)
                    cost_o = cost / (sum(sum(demand)))
                    hops["optimal"].append(cost_o)
                    if i_iter > 0:
                        step, port = cal_change(state_o, adjs["optimal"],
                                                n_node)
                        steps["optimal"].append(step)
                        ports["optimal"].append(port)
                    adjs["optimal"] = state_o
                    print("optimal: {}".format(cost_o))

            if "greedy" in methods:
                if scheme == "bysteps":
                    # origin_graph = nx.from_dict_of_dicts(topo)
                    if i_iter == 0:
                        origin_graph = nx.Graph()
                        origin_graph.add_nodes_from(list(range(n_node)))
                    permatch_new_graph = permatch_model.n_steps_matching(
                        demand, origin_graph, degree, n_steps)
                    state_m = np.array(
                        nx.adjacency_matrix(permatch_new_graph).todense(),
                        np.float32)
                    origin_graph = nx.from_numpy_matrix(state_m)
                if scheme == "complete":
                    state_m = permatch_model.matching(demand, degree)
                cost_m = cal_pathlength(state_m, n_node, demand, degree)
                hops["greedy"].append(cost_m)
                if i_iter > 0:
                    step, port = cal_change(state_m, adjs["greedy"], n_node)
                    steps["greedy"].append(step)
                    ports["greedy"].append(port)
                adjs["greedy"] = state_m
                print("greedy: {}".format(cost_m))

            if "dijgreedy" in methods:
                if scheme == "bysteps":
                    # origin_graph = nx.from_dict_of_dicts(topo)
                    if i_iter == 0:
                        origin_graph = nx.Graph()
                        origin_graph.add_nodes_from(list(range(n_node)))
                    result_graph = dijgreedy_model.topo_nsteps(
                        demand, origin_graph, n_steps)
                    state_d = np.array(
                        nx.adjacency_matrix(result_graph).todense(),
                        np.float32)
                    origin_graph = result_graph
                if scheme == "complete":
                    state_d = dijgreedy_model.topo_scratch(demand, degree)
                cost_d = cal_pathlength(state_d, n_node, demand, degree)
                hops["dijgreedy"].append(cost_d)
                if i_iter > 0:
                    step, port = cal_change(state_d, adjs["dijgreedy"], n_node)
                    steps["dijgreedy"].append(step)
                    ports["dijgreedy"].append(port)
                adjs["dijgreedy"] = state_d
                print("dijkstra greedy: {}".format(cost_d))

            if "param-search" in methods:
                curr_graph = topo
                for i_step in range(n_steps):
                    v = opr.predict(curr_graph, demand)
                    if i_step < n_steps - 1:
                        curr_graph = sim.step_graph(n_node,
                                                    v,
                                                    demand=demand,
                                                    topology=curr_graph,
                                                    allowed_degree=degree)
                    else:
                        cost_s = sim.step(n_node,
                                          v,
                                          demand=demand,
                                          topology=curr_graph,
                                          allowed_degree=degree)
                hops["param-search"].append(cost_s)
                print("search: {}".format(cost_s))

            if "rl" in methods:
                obs = env.reset(demand=demand, degree=degree, provide=True)
                for _ in range(n_steps):
                    action, _ = policy.predict(obs)
                    obs, _, _, _ = env.step(action)
                state_rl = obs2adj(obs, n_node)
                cost_rl = cal_pathlength(state_rl, n_node, demand, degree)
                hops["rl"].append(cost_rl)
                print("RL: {}".format(cost_rl))

            if "oblivious" in methods:
                state_ob = np.zeros((n_node, n_node))
                for i in range(n_node):
                    for j in range(n_node):
                        if (j - i) % n_node == 1 or (j - i) % n_node == 2 or (
                                i - j) % n_node == 1 or (i - j) % n_node == 2:
                            state_ob[i, j] = 1
                cost_ob = cal_pathlength(state_ob, n_node, demand, degree)
                hops["oblivious"].append(cost_ob)
                if i_iter > 0:
                    step, port = cal_change(state_ob, adjs["oblivious"],
                                            n_node)
                    steps["oblivious"].append(step)
                    ports["oblivious"].append(port)
                adjs["oblivious"] = state_ob
                print("oblivious: {}".format(cost_ob))

            ## test supervised learning
            #obs = env.reset(demand=demand,degree=degree,provide=True)
            #demand_input, adj_input, deg_input = obs_process(obs, node_num)
            #potential = model.predict(demand_input,adj_input, deg_input)
            #action = np.squeeze(potential)
            #obs, _, _, _ = env.step(action)
            #state_sl = obs2adj(obs,node_num)
            #cost_sl = cal_pathlength(state_sl, node_num, demand, degree)
            #costs_sl.append(cost_sl)
            #print("SL: {}".format(cost_sl))

            ## optimal (1 step)
            #origin_graph = nx.from_dict_of_dicts(topo)
            #best_action,neigh,opt_graph = opt.compute_optimal(node_num,origin_graph,demand,degree)
            #state_o1 = np.array(nx.adjacency_matrix(opt_graph).todense(), np.float32)
            #cost_o1 = cal_pathlength(state_o1, node_num, demand, degree, 100)
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
            #cost_h = cal_pathlength(state_h, node_num, demand, degree)

    t_end = timer()

    print("Setting:\ndata source = {0}\nn_nodes     = {1}".format(
        data_source, n_node))
    if scheme == "bysteps":
        print("======== Avg_costs & std ({} step(s)) ========".format(n_steps))
    elif scheme == "complete":
        print("========== Avg_costs & std (compl) ===========")

    for m in methods:
        print("[Alg] {}".format(m))
        print("[Average Hop] {}".format(np.mean(hops[m])))
        print("[Standard Deviation Hop] {}".format(np.std(hops[m])))
        print("[Average Step] {}".format(np.mean(steps[m])))
        print("[Standard Deviation Step] {}".format(np.std(steps[m])))
        print("[Average Change Port] {}".format(np.mean(ports[m])))
        print("[Standard Deviation Change Port] {}".format(np.std(ports[m])))
    print("[Average Test Time] {} s".format(t_end - t_begin))


def obs2adj(obs, n_node):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    obs = np.reshape(obs, ((2 * n_node + 1), n_node))
    adj = obs[n_node:-1, :]
    #adj[adj>0] = 1
    return adj


def obs_process(obs, n_node):
    """
    :param obs: N x (N+1) matrix for adjacent matrix with edge features and a vector with node features
    :return: adjacent matrix
    """
    obs = np.reshape(obs, ((2 * n_node + 1), n_node))
    demand = obs[:n_node, :]
    adj = obs[n_node:-1, :]
    deg = obs[-1, :]
    #adj[adj>0] = 1
    return demand[np.newaxis, :], adj[np.newaxis, :], deg[np.newaxis, :]


if __name__ == "__main__":
    main()