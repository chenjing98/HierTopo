import copy
import pickle as pk
import argparse

import numpy as np
import random
import itertools
import networkx as nx

from multiprocessing import Pool
from timeit import default_timer as timer

from .permatch import permatch


class HierTopoPolynAlg(object):

    def __init__(self, n_node, n_degree, n_iter, n_maxstep, k):
        self.n_node = n_node
        self.n_degree = n_degree
        self.n_iter = n_iter
        self.n_maxstep = int(n_node * n_degree / 2)
        self.n_maxstep_force = n_maxstep

        self.k = k

        self.permatch_model = permatch(n_node)

    def apply_policy(self, demand, alpha):
        """
        @param demand: (np.array) N x N
        @param topo: (nx.dict_of_dicts)
        @param alpha: (np.array) N
        @return: hop_cnt: (np.float32) average shortest path length
        """

        G = nx.Graph()
        G.add_nodes_from(list(range(self.n_node)))
        #graph = nx.from_dict_of_dicts(dataset_topo)

        adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
        degree = np.sum(adj, axis=-1)

        z = np.zeros((self.n_node, self.n_node), np.float32)
        for _ in range(self.n_maxstep):
            #x = np.sum(demand, axis=0)
            x = demand / np.max(demand) * 2 - 1  # [N]
            x = x.T
            for i in range(self.n_iter):
                exp_x = self.expand_orders_mat(x)
                weighing_self = np.matmul(exp_x, alpha[0:self.k])
                weighing_neigh = np.matmul(exp_x, alpha[self.k:2 * self.k])
                neighbor_aggr = np.matmul(weighing_neigh, adj)
                g = weighing_self + neighbor_aggr
                #x = g/np.max(g)*2 # N x N
                gpos = np.where(g >= 0, g, z)
                gneg = np.where(g < 0, g, z)
                x = 1 / (1 + np.exp(-gpos)) + np.exp(gneg) / (
                    1 + np.exp(gneg)) - 1 / 2

            v = np.sum(x, axis=0)
            dif = self.cal_diff(v) + 1.0
            degree_full = np.where(degree >= self.n_degree, 1.0, 0.0)
            degree_mask = np.repeat(np.expand_dims(
                degree_full, 0), self.n_node, 0) + np.repeat(
                    np.expand_dims(degree_full, -1), self.n_node, -1)
            mask = adj + np.identity(self.n_node, np.float32) + degree_mask
            masked_dif = (mask == 0) * dif - 1.0
            ind_x, ind_y = np.where(masked_dif == np.max(masked_dif))
            #ind_x, ind_y = np.where(dif==np.max(dif))
            if len(ind_x) < 1:
                continue
            elif len(ind_x) > 1:
                j = random.randint(0, len(ind_x) - 1)
                add_ind = (ind_x[j], ind_y[j])
            else:
                add_ind = (ind_x[0], ind_y[0])

            if (adj[add_ind] != 1) and (degree[add_ind[0]] <
                                        self.n_degree) and (degree[add_ind[1]]
                                                            < self.n_degree):
                G.add_edge(add_ind[0], add_ind[1])
                adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
                degree = np.sum(adj, axis=-1)

        return self.cal_pathlength(demand, G)

    def apply_policy_w_replace(self, demand, alpha):
        """Policy with greedy initialization & link tearing down.
        @param demand: (np.array) N x N
        @param alpha: (np.array) N
        @return: hop_cnt: (np.float32) average shortest path length
        """

        #graph = nx.Graph()
        #graph.add_nodes_from(list(range(n_nodes)))
        #adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
        adj = self.permatch_model.matching(
            demand,
            np.ones((self.n_node, )) * (self.n_degree - 1))
        G = nx.from_numpy_matrix(adj)
        degree = np.sum(adj, axis=-1)

        z = np.zeros((self.n_node, self.n_node), np.float32)
        for s in range(self.n_maxstep_force):
            x = demand / np.max(demand) * 2 - 1  # [N]
            x = x.T
            for i in range(self.n_iter):
                exp_x = self.expand_orders_mat(x)
                weighing_self = np.matmul(
                    exp_x, alpha[2 * i * self.k:(2 * i + 1) * self.k])
                weighing_neigh = np.matmul(
                    exp_x, alpha[(2 * i + 1) * self.k:(2 * i + 2) * self.k])
                neighbor_aggr = np.matmul(weighing_neigh, adj)
                g = weighing_self + neighbor_aggr
                #x = g/np.max(g)*2 # N x N
                gpos = np.where(g >= 0, g, z)
                gneg = np.where(g < 0, g, z)
                x = 1 / (1 + np.exp(-gpos)) + np.exp(gneg) / (
                    1 + np.exp(gneg)) - 1 / 2

            v = np.sum(x, axis=0)
            dif = self.cal_diff(v) + 1.0
            mask = adj + np.identity(self.n_node, np.float32)
            masked_dif = (mask == 0) * dif - 1.0
            ind_x, ind_y = np.where(masked_dif == np.max(masked_dif))
            if len(ind_x) < 1:
                continue
            elif len(ind_x) > 1:
                j = random.randint(0, len(ind_x) - 1)
                add_ind = (ind_x[j], ind_y[j])
            else:
                add_ind = (ind_x[0], ind_y[0])
            #if add_ind[0] == add_ind[1] or adj[add_ind] == 1:
            #    print("wrong in the find")

            rm_inds = []
            loss = 0
            if (degree[add_ind[0]] >= self.n_degree):
                dif_at_n0 = np.max(dif) + 1.0 - dif[add_ind[0]]
                dif_n0_masked = np.multiply(adj[add_ind[0]], dif_at_n0)
                loss += np.max(dif) + 1.0 - np.max(dif_n0_masked)
                if loss > np.max(masked_dif):
                    #print("Stop at No.{} step".format(s))
                    break
                rm_ind = np.where(dif_n0_masked == np.max(dif_n0_masked))[0][0]
                #if not graph.has_edge(add_ind[0],rm_ind):
                #    print("wrong at first remove")
                G.remove_edge(add_ind[0], rm_ind)
                rm_inds.append((add_ind[0], rm_ind))
            if (degree[add_ind[1]] >= self.n_degree):
                dif_at_n1 = np.max(dif) + 1.0 - dif[add_ind[1]]
                dif_n1_masked = np.multiply(adj[add_ind[1]], dif_at_n1)
                loss += np.max(dif) + 1.0 - np.max(dif_n1_masked)
                if loss > np.max(masked_dif):
                    for removed in rm_inds:
                        G.add_edge(removed[0], removed[1])
                    #print("Stop at No.{} step".format(s))
                    break
                rm_ind = np.where(dif_n1_masked == np.max(dif_n1_masked))[0][0]
                #if not graph.has_edge(add_ind[1],rm_ind):
                #    print("wrong at second remove")
                G.remove_edge(add_ind[1], rm_ind)

            if (degree[add_ind[0]] < self.n_degree) and (degree[add_ind[1]] <
                                                         self.n_degree):
                G.add_edge(add_ind[0], add_ind[1])
            adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
            degree = np.sum(adj, axis=-1)

        if s == self.n_maxstep_force - 1:
            print(
                "Maximum adjustement step ({}) reach. Unwillingly terminated.".
                format(self.n_maxstep_force))

        return self.cal_pathlength(demand, G)

    def apply_policy_w_replace_nsquare_list(self, demand, alpha):
        G = nx.Graph()
        G.add_nodes_from(list(range(self.n_node)))
        adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
        #adj = permatch_model.matching(demand, np.ones((n_node,)) * (n_degree-1))
        #graph = nx.from_numpy_matrix(adj)
        degree = np.sum(adj, axis=-1)

        remaining_choices = []
        for i in range(self.n_node - 1):
            for j in range(i + 1, self.n_node):
                remaining_choices.append(i * self.n_node + j)
        rm_inds = []
        failed_attempts = []

        v = self.cal_v(demand, alpha, adj)
        dif_e = self.cal_diff_in_range(v, remaining_choices)
        while remaining_choices:
            curr_e_num = dif_e.index(max(dif_e))
            curr_e = remaining_choices[curr_e_num]
            v1 = int(curr_e / self.n_node)
            v2 = curr_e % self.n_node
            if adj[v1, v2] == 1:
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
                continue
            if degree[v1] < self.n_degree and degree[v2] < self.n_degree:
                G.add_edge(v1, v2)
                adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
                degree = np.sum(adj, axis=-1)
                v = self.cal_v(demand, alpha, adj)
                del remaining_choices[curr_e_num]
                dif_e = self.cal_diff_in_range(v, remaining_choices)
                continue
            if len(failed_attempts) > 20:
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
                continue
            # need to remove some edges
            if degree[v1] >= self.n_degree and degree[v2] >= self.n_degree:
                v1_neighbor = [n for n in G.neighbors(v1)]
                v1_edges = np.where(
                    np.array(v1_neighbor) > v1,
                    v1 * self.n_node + np.array(v1_neighbor),
                    np.array(v1_neighbor) * self.n_node + v1).tolist()
                dif_v1 = self.cal_diff_in_range(v, v1_edges)
                v1_e_num = dif_v1.index(min(dif_v1))
                e1_rm = v1_edges[v1_e_num]

                v2_neighbor = [n for n in G.neighbors(v2)]
                v2_edges = np.where(
                    np.array(v2_neighbor) > v2,
                    v2 * self.n_node + np.array(v2_neighbor),
                    np.array(v2_neighbor) * self.n_node + v2).tolist()
                dif_v2 = self.cal_diff_in_range(v, v2_edges)
                v2_e_num = dif_v2.index(min(dif_v2))
                e2_rm = v2_edges[v2_e_num]

                rm_inds = [e1_rm, e2_rm]
                adj_rp = copy.deepcopy(adj)
                adj_rp[int(e1_rm / self.n_node), e1_rm % self.n_node] = 0
                adj_rp[e1_rm % self.n_node, int(e1_rm / self.n_node)] = 0
                adj_rp[int(e2_rm / self.n_node), e2_rm % self.n_node] = 0
                adj_rp[e2_rm % self.n_node, int(e2_rm / self.n_node)] = 0
                adj_rp[v1, v2] = 1
                adj_rp[v2, v1] = 1
                v_rp = self.cal_v(demand, alpha, adj_rp)
                if max(dif_e) + sum(self.cal_diff_in_range(v, rm_inds)) > sum(
                        self.cal_diff_in_range(v_rp, [curr_e])) + sum(
                            self.cal_diff_in_range(v_rp, rm_inds)):
                    G.remove_edge(int(e1_rm / self.n_node),
                                  e1_rm % self.n_node)
                    G.remove_edge(int(e2_rm / self.n_node),
                                  e2_rm % self.n_node)
                    G.add_edge(v1, v2)
                    adj = adj_rp
                    degree = np.sum(adj, axis=-1)
                    v = v_rp
                    del remaining_choices[curr_e_num]
                    dif_e = self.cal_diff_in_range(v, remaining_choices)
                else:
                    failed_attempts.append(curr_e)
                    del remaining_choices[curr_e_num]
                    del dif_e[curr_e_num]
            elif degree[v1] >= self.n_degree:
                v1_neighbor = [n for n in G.neighbors(v1)]
                v1_edges = np.where(
                    np.array(v1_neighbor) > v1,
                    v1 * self.n_node + np.array(v1_neighbor),
                    np.array(v1_neighbor) * self.n_node + v1).tolist()
                dif_v1 = self.cal_diff_in_range(v, v1_edges)
                v1_e_num = dif_v1.index(min(dif_v1))
                e1_rm = v1_edges[v1_e_num]
                rm_inds.append(e1_rm)
                adj_rp = copy.deepcopy(adj)
                adj_rp[int(e1_rm / self.n_node), e1_rm % self.n_node] = 0
                adj_rp[e1_rm % self.n_node, int(e1_rm / self.n_node)] = 0
                adj_rp[v1, v2] = 1
                adj_rp[v2, v1] = 1
                v_rp = self.cal_v(demand, alpha, adj_rp)
                if max(dif_e) + sum(self.cal_diff_in_range(v, rm_inds)) > sum(
                        self.cal_diff_in_range(v_rp, [curr_e])) + sum(
                            self.cal_diff_in_range(v_rp, rm_inds)):
                    G.remove_edge(int(e1_rm / self.n_node),
                                  e1_rm % self.n_node)
                    G.add_edge(v1, v2)
                    adj = adj_rp
                    degree = np.sum(adj, axis=-1)
                    v = v_rp
                    del remaining_choices[curr_e_num]
                    dif_e = self.cal_diff_in_range(v, remaining_choices)
                    rm_inds = []
                else:
                    failed_attempts.append(curr_e)
                    del remaining_choices[curr_e_num]
                    del dif_e[curr_e_num]
            else:
                v2_neighbor = [n for n in G.neighbors(v2)]
                v2_edges = np.where(
                    np.array(v2_neighbor) > v2,
                    v2 * self.n_node + np.array(v2_neighbor),
                    np.array(v2_neighbor) * self.n_node + v2).tolist()
                dif_v2 = self.cal_diff_in_range(v, v2_edges)
                v2_e_num = dif_v2.index(min(dif_v2))
                e2_rm = v2_edges[v2_e_num]
                rm_inds.append(e2_rm)
                adj_rp = copy.deepcopy(adj)
                adj_rp[int(e2_rm / self.n_node), e2_rm % self.n_node] = 0
                adj_rp[e2_rm % self.n_node, int(e2_rm / self.n_node)] = 0
                adj_rp[v1, v2] = 1
                adj_rp[v2, v1] = 1
                v_rp = self.cal_v(demand, alpha, adj_rp)
                if max(dif_e) + sum(self.cal_diff_in_range(v, rm_inds)) > sum(
                        self.cal_diff_in_range(v_rp, [curr_e])) + sum(
                            self.cal_diff_in_range(v_rp, rm_inds)):
                    G.remove_edge(int(e2_rm / self.n_node),
                                  e2_rm % self.n_node)
                    G.add_edge(v1, v2)
                    adj = adj_rp
                    degree = np.sum(adj, axis=-1)
                    v = v_rp
                    del remaining_choices[curr_e_num]
                    dif_e = self.cal_diff_in_range(v, remaining_choices)
                    rm_inds = []
                else:
                    failed_attempts.append(curr_e)
                    del remaining_choices[curr_e_num]
                    del dif_e[curr_e_num]
        #print(graph.number_of_edges())

        return self.cal_pathlength(demand, G)

    def apply_policy_w_replace_run(self, params):
        demand = params["demand"]
        alpha = params["alpha"]
        G = nx.Graph()
        G.add_nodes_from(list(range(self.n_node)))
        adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
        #adj = permatch_model.matching(demand, np.ones((n_node,)) * (n_degree-1))
        #graph = nx.from_numpy_matrix(adj)
        degree = np.sum(adj, axis=-1)

        remaining_choices = []
        for i in range(self.n_node - 1):
            for j in range(i + 1, self.n_node):
                remaining_choices.append(i * self.n_node + j)
        rm_inds = []
        failed_attempts = []

        v = self.cal_v(demand, alpha, adj)
        dif_e = self.cal_diff_in_range(v, remaining_choices)
        while remaining_choices:
            curr_e_num = dif_e.index(max(dif_e))
            curr_e = remaining_choices[curr_e_num]
            v1 = int(curr_e / self.n_node)
            v2 = curr_e % self.n_node
            if adj[v1, v2] == 1:
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
                continue
            if degree[v1] < self.n_degree and degree[v2] < self.n_degree:
                G.add_edge(v1, v2)
                adj = np.array(nx.adjacency_matrix(G).todense(), np.float32)
                degree = np.sum(adj, axis=-1)
                v = self.cal_v(demand, alpha, adj)
                del remaining_choices[curr_e_num]
                dif_e = self.cal_diff_in_range(v, remaining_choices)
                continue
            if len(failed_attempts) > 20:
                del remaining_choices[curr_e_num]
                del dif_e[curr_e_num]
                continue
            # need to remove some edges
            if degree[v1] >= self.n_degree and degree[v2] >= self.n_degree:
                v1_neighbor = [n for n in G.neighbors(v1)]
                v1_edges = np.where(
                    np.array(v1_neighbor) > v1,
                    v1 * self.n_node + np.array(v1_neighbor),
                    np.array(v1_neighbor) * self.n_node + v1).tolist()
                dif_v1 = self.cal_diff_in_range(v, v1_edges)
                v1_e_num = dif_v1.index(min(dif_v1))
                e1_rm = v1_edges[v1_e_num]

                v2_neighbor = [n for n in G.neighbors(v2)]
                v2_edges = np.where(
                    np.array(v2_neighbor) > v2,
                    v2 * self.n_node + np.array(v2_neighbor),
                    np.array(v2_neighbor) * self.n_node + v2).tolist()
                dif_v2 = self.cal_diff_in_range(v, v2_edges)
                v2_e_num = dif_v2.index(min(dif_v2))
                e2_rm = v2_edges[v2_e_num]

                rm_inds = [e1_rm, e2_rm]
                adj_rp = copy.deepcopy(adj)
                adj_rp[int(e1_rm / self.n_node), e1_rm % self.n_node] = 0
                adj_rp[e1_rm % self.n_node, int(e1_rm / self.n_node)] = 0
                adj_rp[int(e2_rm / self.n_node), e2_rm % self.n_node] = 0
                adj_rp[e2_rm % self.n_node, int(e2_rm / self.n_node)] = 0
                adj_rp[v1, v2] = 1
                adj_rp[v2, v1] = 1
                v_rp = self.cal_v(demand, alpha, adj_rp)
                if max(dif_e) + sum(self.cal_diff_in_range(v, rm_inds)) > sum(
                        self.cal_diff_in_range(v_rp, [curr_e])) + sum(
                            self.cal_diff_in_range(v_rp, rm_inds)):
                    G.remove_edge(int(e1_rm / self.n_node),
                                  e1_rm % self.n_node)
                    G.remove_edge(int(e2_rm / self.n_node),
                                  e2_rm % self.n_node)
                    G.add_edge(v1, v2)
                    adj = adj_rp
                    degree = np.sum(adj, axis=-1)
                    v = v_rp
                    del remaining_choices[curr_e_num]
                    dif_e = self.cal_diff_in_range(v, remaining_choices)
                else:
                    failed_attempts.append(curr_e)
                    del remaining_choices[curr_e_num]
                    del dif_e[curr_e_num]
            elif degree[v1] >= self.n_degree:
                v1_neighbor = [n for n in G.neighbors(v1)]
                v1_edges = np.where(
                    np.array(v1_neighbor) > v1,
                    v1 * self.n_node + np.array(v1_neighbor),
                    np.array(v1_neighbor) * self.n_node + v1).tolist()
                dif_v1 = self.cal_diff_in_range(v, v1_edges)
                v1_e_num = dif_v1.index(min(dif_v1))
                e1_rm = v1_edges[v1_e_num]
                rm_inds.append(e1_rm)
                adj_rp = copy.deepcopy(adj)
                adj_rp[int(e1_rm / self.n_node), e1_rm % self.n_node] = 0
                adj_rp[e1_rm % self.n_node, int(e1_rm / self.n_node)] = 0
                adj_rp[v1, v2] = 1
                adj_rp[v2, v1] = 1
                v_rp = self.cal_v(demand, alpha, adj_rp)
                if max(dif_e) + sum(self.cal_diff_in_range(v, rm_inds)) > sum(
                        self.cal_diff_in_range(v_rp, [curr_e])) + sum(
                            self.cal_diff_in_range(v_rp, rm_inds)):
                    G.remove_edge(int(e1_rm / self.n_node),
                                  e1_rm % self.n_node)
                    G.add_edge(v1, v2)
                    adj = adj_rp
                    degree = np.sum(adj, axis=-1)
                    v = v_rp
                    del remaining_choices[curr_e_num]
                    dif_e = self.cal_diff_in_range(v, remaining_choices)
                    rm_inds = []
                else:
                    failed_attempts.append(curr_e)
                    del remaining_choices[curr_e_num]
                    del dif_e[curr_e_num]
            else:
                v2_neighbor = [n for n in G.neighbors(v2)]
                v2_edges = np.where(
                    np.array(v2_neighbor) > v2,
                    v2 * self.n_node + np.array(v2_neighbor),
                    np.array(v2_neighbor) * self.n_node + v2).tolist()
                dif_v2 = self.cal_diff_in_range(v, v2_edges)
                v2_e_num = dif_v2.index(min(dif_v2))
                e2_rm = v2_edges[v2_e_num]
                rm_inds.append(e2_rm)
                adj_rp = copy.deepcopy(adj)
                adj_rp[int(e2_rm / self.n_node), e2_rm % self.n_node] = 0
                adj_rp[e2_rm % self.n_node, int(e2_rm / self.n_node)] = 0
                adj_rp[v1, v2] = 1
                adj_rp[v2, v1] = 1
                v_rp = self.cal_v(demand, alpha, adj_rp)
                if max(dif_e) + sum(self.cal_diff_in_range(v, rm_inds)) > sum(
                        self.cal_diff_in_range(v_rp, [curr_e])) + sum(
                            self.cal_diff_in_range(v_rp, rm_inds)):
                    G.remove_edge(int(e2_rm / self.n_node),
                                  e2_rm % self.n_node)
                    G.add_edge(v1, v2)
                    adj = adj_rp
                    degree = np.sum(adj, axis=-1)
                    v = v_rp
                    del remaining_choices[curr_e_num]
                    dif_e = self.cal_diff_in_range(v, remaining_choices)
                    rm_inds = []
                else:
                    failed_attempts.append(curr_e)
                    del remaining_choices[curr_e_num]
                    del dif_e[curr_e_num]
        #print(graph.number_of_edges())
        return G

    def single_move_wo_replace(self, demand, graph, cand, alpha):
        if len(cand) == 0:
            return True, 0, cand
        adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
        cand_r = copy.deepcopy(cand)
        v = self.cal_v(demand, alpha, adj)
        dif_e = self.cal_diff_in_range(v, cand_r)
        e_idx = dif_e.index(max(dif_e))
        e = cand_r[e_idx]
        n = self.edge_to_node(e)
        n0 = n[0]
        n1 = n[1]
        while True:
            if graph.degree(n0) < self.n_degree and graph.degree(
                    n1) < self.n_degree:
                return False, e, cand_r
            del cand_r[e_idx]
            if len(cand_r) == 0:
                return True, 0, cand_r
            dif_e = self.cal_diff_in_range(v, cand_r)
            e_idx = dif_e.index(max(dif_e))
            e = cand_r[e_idx]
            n = self.edge_to_node(e)
            n0 = n[0]
            n1 = n[1]

    def single_move_w_replace(self, demand, graph, cand, alpha):
        if len(cand) == 0:
            return True, 0, [], cand
        adj = np.array(nx.adjacency_matrix(graph).todense(), np.float32)
        cand_r = copy.deepcopy(cand)
        v = self.cal_v(demand, alpha, adj)
        dif_e = self.cal_diff_in_range(v, cand_r)
        e_idx = dif_e.index(max(dif_e))
        e = cand_r[e_idx]
        n = self.edge_to_node(e)
        n0 = n[0]
        n1 = n[1]

        while True:
            if graph.degree(n0) < self.n_degree and graph.degree(
                    n1) < self.n_degree:
                return False, e, [], cand_r
            adj_rp = copy.deepcopy(adj)
            edge_rm = []
            if graph.degree(n0) >= self.n_degree:
                e1_rm = self.find_nbr_rm_cand(graph, n0, v)
                adj_rp[int(e1_rm / self.n_node), int(e1_rm % self.n_node)] = 0
                adj_rp[int(e1_rm % self.n_node), int(e1_rm / self.n_node)] = 0
                edge_rm.append(e1_rm)
            if graph.degree(n1) >= self.n_degree:
                e2_rm = self.find_nbr_rm_cand(graph, n1, v)
                adj_rp[int(e2_rm / self.n_node), int(e2_rm % self.n_node)] = 0
                adj_rp[int(e2_rm % self.n_node), int(e2_rm / self.n_node)] = 0
                edge_rm.append(e2_rm)

            adj_rp[n0, n1] = 1
            adj_rp[n1, n0] = 1
            v_rp = self.cal_v(demand, alpha, adj_rp)
            if max(dif_e) + sum(self.cal_diff_in_range(v, edge_rm)) > sum(
                    self.cal_diff_in_range(v_rp, [e])) + sum(
                        self.cal_diff_in_range(v_rp, edge_rm)):
                return False, e, edge_rm, cand_r
            del cand_r[e_idx]
            if len(cand_r) == 0:
                return True, 0, [], cand_r
            dif_e = self.cal_diff_in_range(v, cand_r)
            e_idx = dif_e.index(max(dif_e))
            e = cand_r[e_idx]
            n = self.edge_to_node(e)
            n0 = n[0]
            n1 = n[1]

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

    def expand_orders_mat(self, feature):
        """
        :param feature: (np.array) N x N
        :return exp_feature: (np.array) N x N x k
        """
        N = feature.shape[0]
        exp_feature = np.zeros((N, N, self.k), np.float32)
        for i in range(self.k):
            exp_feature[:, :, i] = np.power(feature, i)
        return exp_feature

    def find_nbr_rm_cand(self, graph, work_node, v):
        nbr_node = [n for n in graph.neighbors(work_node)]
        nbr_edge = np.where(
            np.array(nbr_node) > work_node,
            work_node * self.n_node + np.array(nbr_node),
            np.array(nbr_node) * self.n_node + work_node).tolist()
        dif_rm = self.cal_diff_in_range(v, nbr_edge)
        e_idx_rm = dif_rm.index(min(dif_rm))
        e_rm = nbr_edge[e_idx_rm]

        return e_rm

    def cal_diff(self, v):
        N = v.shape[0]
        dif = np.repeat(np.expand_dims(v, 0), N, 0) - np.repeat(
            np.expand_dims(v, -1), N, -1)
        dif = np.abs(dif)
        return dif

    def cal_diff_in_range(self, v, edges):
        dif = []
        for i in range(len(edges)):
            e = edges[i]
            v1 = int(e / self.n_node)
            v2 = e % self.n_node
            dif.append(np.abs(v[v1] - v[v2]))
        return dif

    def cal_v(self, demand, alpha, adj):
        x = demand / np.max(demand) * 2 - 1  # [N]
        x = x.T
        z = np.zeros((self.n_node, self.n_node), np.float32)
        for i in range(self.n_iter):
            exp_x = self.expand_orders_mat(x)
            weighing_self = np.matmul(exp_x, alpha[0:self.k])
            weighing_neigh = np.matmul(exp_x, alpha[self.k:2 * self.k])
            neighbor_aggr = np.matmul(weighing_neigh, adj)
            g = weighing_self + neighbor_aggr
            #x = g/np.max(g)*2 # N x N
            gpos = np.where(g >= 0, g, z)
            gneg = np.where(g < 0, g, z)
            x = 1 / (1 + np.exp(-gpos)) + np.exp(gneg) / (1 +
                                                          np.exp(gneg)) - 1 / 2

        v = np.sum(x, axis=0)
        return v

    def edge_to_node(self, e):
        v1 = np.floor(e / self.n_node)
        v2 = e - v1 * self.n_node
        return [v1, v2]


def test_robust(solution, test_size, dataset, n_node, n_degree, n_iter,
                n_maxstep, k, ad_scheme):
    metrics = []

    hiertopo = HierTopoPolynAlg(n_node, n_degree, n_iter, n_maxstep, k)

    for i in range(test_size):
        if ad_scheme == "add":
            m = hiertopo.apply_policy(dataset[i], solution)
        else:
            m = hiertopo.apply_policy_w_replace_nsquare_list(
                dataset[i], solution)
        metrics.append(m)
        #print("[No. {0}] {1}".format(i,m))
    output = np.mean(metrics)
    output_std = np.std(metrics)
    return output, output_std


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

    hiertopo = HierTopoPolynAlg(n_node, n_degree, n_iter, n_maxstep, k)

    pool = Pool()
    graphs = pool.map(hiertopo.apply_policy_w_replace_run, params)
    pool.close()
    pool.join()

    t1 = timer()
    print("Decision Time {}".format(t1 - t0))

    for i in range(test_size):
        m = hiertopo.cal_pathlength(dataset[i], graphs[i])
        metrics.append(m)
    output = np.mean(metrics)
    output_std = np.std(metrics)
    return output, output_std


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

    ad_scheme = args.ad_scheme
    data_source = args.data_source

    # ============ Setup testing datasets ============
    if data_source == "random":
        n_node = 8
        n_degree = 4
        n_testings = 1000
        file_demand_degree = '../../data/10000_8_4_test.pk3'
        file_topo = "../../data/10000_8_4_topo_test.pk3"
    elif data_source == "nsfnet":
        n_node = 14
        n_degree = 4
        n_testings = 100
        file_demand_degree = '../../data/nsfnet/demand_100.pkl'
        file_topo = '../../data/nsfnet/topology.pkl'
    elif data_source == "geant2":
        n_node = 24
        n_degree = 8
        n_testings = 100
        file_demand_degree = '../../data/geant2/demand_100.pkl'
        file_topo = '../../data/geant2/topology.pkl'
    elif data_source == "scratch":
        n_node = n_node
        n_testings = 1000
        #n_iter = int(n_nodes*(n_nodes-1)/2)
        file_demand = '../../data/2000_{0}_{1}_logistic.pk3'.format(
            n_node, n_degree)
    else:
        print("data_source {} unrecognized.".format(data_source))
        exit(1)

    file_logging = '../../poly_log/log{0}_{1}_{2}_{3}_same.pkl'.format(
        n_nodes_param, n_degree, k, n_iters_param)
    if ad_scheme == "replace":
        file_logging = '../../poly_log/log{0}_{1}_{2}_{3}_same_repl.pkl'.format(
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

    t_end = timer()
    print(
        "Prediction = {0}, std = {1}, test_time for {2} samples = {3}s".format(
            pred, pred_std, n_testings, t_end - t_begin))


if __name__ == "__main__":
    main()