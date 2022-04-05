import argparse
import pickle as pk

import numpy as np
import itertools
import networkx as nx

from multiprocessing import Pool
from timeit import default_timer as timer

from polyfit.hiertopo import HierTopoPolynAlg
from baseline.dijkstra_greedy import DijGreedyAlg


class SafeHierTopoAlg(object):

    def __init__(self, n_node, n_degree, n_iter, n_maxstep, k):
        self.n_node = n_node
        self.n_degree = n_degree

        self.hiertopo_model = HierTopoPolynAlg(n_node, n_degree, n_iter,
                                               n_maxstep, k)
        self.rgreedy_model = DijGreedyAlg(n_node, n_degree)

        self.cntr = 0
        self.period = 1
        self.step = 0

    def single_move(self, demand, graph, cand_ht, cand_rg, alpha, is_verbose):
        is_end_ht, e_ht, cand_ht_m = self.hiertopo_model.single_move_wo_replace(
            demand, graph, cand_ht, alpha)
        is_end_rg, e_rg, cand_rg_m = self.rgreedy_model.single_move_wo_replace(
            demand, graph, cand_rg)

        if is_verbose:
            print("[Step {0}] [HierTopo] end {1}, edge {2}, candidate {3}".
                  format(self.step, is_end_ht, e_ht, cand_ht_m))
            print(
                "[Step {0}] [RGreedy] end {1}, edge {2}, candidate {3}".format(
                    self.step, is_end_rg, e_rg, cand_rg_m))

        is_end, e, cand_ht_new, cand_rg_new = self.fallback(
            is_end_ht, e_ht, cand_ht_m, is_end_rg, e_rg, cand_rg_m)

        if is_verbose:
            print(
                "[Step {0}] [Safe] end {1}, edge {2}, candidate {3}, candidate {4}"
                .format(self.step, is_end, e, cand_ht_new, cand_rg_new))

        if not is_end:
            n = self.hiertopo_model.edge_to_node(e)
            graph.add_edge(n[0], n[1])
            if is_verbose:
                print("[Step {0}] Action: ({1}, {2})".format(
                    self.step, n[0], n[1]))
            self.step += 1

        return is_end, graph, cand_ht_new, cand_rg_new

    def fallback(self, is_end_ht, e_ht, cand_ht, is_end_rg, e_rg, cand_rg):
        return self.fallback_period(is_end_ht, e_ht, cand_ht, is_end_rg, e_rg,
                                    cand_rg)

    def fallback_period(self, is_end_ht, e_ht, cand_ht, is_end_rg, e_rg,
                        cand_rg):
        if is_end_ht and is_end_rg:
            return True, 0, cand_ht, cand_rg
        if is_end_ht:
            return False, e_rg, cand_ht, cand_rg
        if is_end_rg:
            return False, e_ht, cand_ht, cand_rg

        # both algorithm has normal output
        if self.cntr % self.period == 0:
            # use Hiertopo's decision
            if e_ht in cand_rg:
                e_idx = cand_rg.index(e_ht)
                del cand_rg[e_idx]
            if e_ht in cand_ht:
                e_idx = cand_ht.index(e_ht)
                del cand_ht[e_idx]
            return False, e_ht, cand_ht, cand_rg
        else:
            # use routing-greedy decision:
            if e_rg in cand_rg:
                e_idx = cand_rg.index(e_rg)
                del cand_rg[e_idx]
            if e_rg in cand_ht:
                e_idx = cand_ht.index(e_rg)
                del cand_ht[e_idx]
            return False, e_rg, cand_ht, cand_rg

    def run(self, params, is_verbose=False):
        demand = params["demand"]
        alpha = params["alpha"]
        G = nx.Graph()
        G.add_nodes_from(list(range(self.n_node)))
        adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
        #adj = permatch_model.matching(demand, np.ones((n_node,)) * (n_degree-1))
        #graph = nx.from_numpy_matrix(adj)
        degree = np.sum(adj, axis=-1)

        cand_ht = []
        cand_rg = []
        for i in range(self.n_node - 1):
            for j in range(i + 1, self.n_node):
                cand_ht.append(i * self.n_node + j)
                cand_rg.append(i * self.n_node + j)

        is_end = False
        while not is_end:
            is_end, G, cand_ht, cand_rg = self.single_move(
                demand, G, cand_ht, cand_rg, alpha, is_verbose)

        return G

    def cal_pathlength(self, demand, graph):
        n_node = demand.shape[0]
        score = 0
        for s, d in itertools.product(range(n_node), range(n_node)):
            try:
                cur_path_length = float(
                    nx.shortest_path_length(graph, source=s, target=d))
            except nx.exception.NetworkXNoPath:
                cur_path_length = float(n_node)

            score += cur_path_length * demand[s, d]
        score /= (sum(sum(demand)))
        return score


def test_mp(solution, test_size, dataset, n_node, n_degree, n_iter, n_maxstep,
            k):
    # Run the test parallelly
    params = []
    metrics = []

    t0 = timer()

    for i in range(test_size):
        param = {}
        param["demand"] = dataset[i]
        param["alpha"] = solution
        params.append(param)

    safe_model = SafeHierTopoAlg(n_node, n_degree, n_iter, n_maxstep, k)

    pool = Pool()
    graphs = pool.map(safe_model.run, params)
    pool.close()
    pool.join()

    t1 = timer()
    print("Decision Time {}".format(t1 - t0))

    for i in range(test_size):
        m = safe_model.cal_pathlength(dataset[i], graphs[i])
        metrics.append(m)
    output = np.mean(metrics)
    output_std = np.std(metrics)
    return output, output_std


def test_standalone(solution, n_data, dataset, n_node, n_degree, n_iter,
                    n_maxstep, k, is_verbose):
    param = {}
    if n_data < len(dataset):
        param["demand"] = dataset[n_data]
    else:
        param["demand"] = dataset[0]
    param["alpha"] = solution

    if is_verbose:
        print("Dataset loaded. Ready to run.")
    safe_model = SafeHierTopoAlg(n_node, n_degree, n_iter, n_maxstep, k)
    G = safe_model.run(param, True)
    h = safe_model.cal_pathlength(param["demand"], G)
    return h, 0


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-n",
                        "--n_node",
                        type=int,
                        help="Number of nodes in the network",
                        default=30)
    parser.add_argument("-d",
                        "--n_degree",
                        type=int,
                        help="Degree limit for each node",
                        default=4)
    parser.add_argument("-i",
                        "--n_iter",
                        type=int,
                        help="Number of iterations",
                        default=14)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument(
        "-np",
        "--n_node_param",
        type=int,
        help="Number of nodes in the network when the GNN model is trained",
        default=10)
    parser.add_argument(
        "-ip",
        "--n_iter_param",
        type=int,
        help="Number of iterations when the GNN model is trained",
        default=10)
    parser.add_argument(
        "-s",
        "--max_step",
        type=int,
        help="Maximum number of steps before topology adjustement ends",
        default=20)
    parser.add_argument("-k",
                        type=int,
                        help="Order of the local policy polynomial",
                        default=3)
    parser.add_argument("-ds",
                        "--data_source",
                        type=str,
                        help="The source of dataset",
                        default="scratch",
                        choices=["random", "nsfnet", "geant2", "scratch"])
    parser.add_argument("-a",
                        "--ad_scheme",
                        type=str,
                        help="The topology adjustment scheme",
                        default="replace",
                        choices=["replace", "add"])
    args = parser.parse_args()

    k = args.k
    n_node = args.n_node
    n_nodes_param = args.n_node_param
    n_iter = args.n_iter
    n_iters_param = args.n_iter_param
    n_degree = args.n_degree
    n_maxstep = args.max_step
    is_verbose = args.verbose

    ad_scheme = args.ad_scheme
    data_source = args.data_source

    # ============ Setup testing datasets ============
    if data_source == "random":
        n_node = 8
        n_degree = 4
        n_testings = 1000
        file_demand_degree = '../data/10000_8_4_test.pk3'
        file_topo = "../data/10000_8_4_topo_test.pk3"
    elif data_source == "nsfnet":
        n_node = 14
        n_degree = 4
        n_testings = 100
        file_demand_degree = '../data/nsfnet/demand_100.pkl'
        file_topo = '../data/nsfnet/topology.pkl'
    elif data_source == "geant2":
        n_node = 24
        n_degree = 8
        n_testings = 100
        file_demand_degree = '../data/geant2/demand_100.pkl'
        file_topo = '../data/geant2/topology.pkl'
    elif data_source == "scratch":
        n_node = n_node
        n_testings = 1000
        #n_iter = int(n_nodes*(n_nodes-1)/2)
        file_demand = '../data/2000_{0}_{1}_logistic.pk3'.format(
            n_node, n_degree)
    else:
        print("data_source {} unrecognized.".format(data_source))
        exit(1)

    file_logging = '../poly_log/log{0}_{1}_{2}_{3}_same.pkl'.format(
        n_nodes_param, n_degree, k, n_iters_param)
    if ad_scheme == "replace":
        file_logging = '../poly_log/log{0}_{1}_{2}_{3}_same_repl.pkl'.format(
            n_nodes_param, n_degree, k, n_iters_param)
    with open(file_demand, 'rb') as f1:
        dataset = pk.load(f1)
    #with open(file_topo, 'rb') as f2:
    #    dataset_topo = pk.load(f2)
    with open(file_logging, 'rb') as f3:
        solution = pk.load(f3)["solution"]

    # ============ Start testing ============
    t_begin = timer()

    pred, pred_std = test_mp(solution, n_testings, dataset, n_node, n_degree,
                             n_iter, n_maxstep, k)

    # pred, pred_std = test_standalone(solution, 0, dataset, n_node, n_degree, n_iter, n_maxstep, k, is_verbose)

    t_end = timer()
    print(
        "Prediction = {0}, std = {1}, test_time for {2} samples = {3}s".format(
            pred, pred_std, n_testings, t_end - t_begin))


if __name__ == "__main__":
    main()