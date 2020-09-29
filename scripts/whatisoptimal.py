import networkx as nx
import numpy as np
import itertools
import copy
from multiprocessing import Pool

class optimal(object):
    def __init__(self):
        self.inf = 1e6

    def compute_optimal(self,n_nodes,graph,demand,allowed_degree):
        """
        Args:
            n_nodes: the total number of nodes in the network
            graph: networkx object, current topology
            demand: traffic demands in matrix form
            allowed_degree: a vector for degree constraints
        Returns:
            optimal actions: the edge between a pair of nodes to add
            optimal V: in order to pretrain the RL model supervisedly
        """
        self.n_nodes = n_nodes
        self.graph = copy.deepcopy(graph)
        self.demand = demand
        self.allowed_degree = allowed_degree
        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        self.origin_cost = self.cal_cost()
        
        costs = []
        neighbors = []
        for i in range(n_nodes-1):
            for j in range(i+1,n_nodes):
                if graph.has_edge(i,j):
                    # already has this edge
                    costs.append(self.inf)
                    neighbors.append(dict())
                else:
                    # consider how good it is to add edge (i,j)
                    cost, neigh = self.consideration(i,j)
                    costs.append(cost)
                    neighbors.append(neigh)
        if min(costs) < self.origin_cost:
            ind = costs.index(min(costs))
            neigh = neighbors[ind]
            bestaction = edge_to_node(n_nodes,ind)
            v1 = bestaction[0]
            v2 = bestaction[1]
            self._add_edge(v1,v2)
            if "left" in neigh:
                self._remove_edge(neigh["left"],v1)
            if "right" in neigh:
                self._remove_edge(neigh["right"],v2)

            return bestaction, neigh, self.graph
        else:
            return [], dict(), graph

    def multistep_DFS(self, n_nodes, graph, demand, allowed_degree, n_steps=1):
        """
        :param n_nodes: (int) the total number of nodes in the network
        :param graph: (nxdict) current topology
        :param demand: (npmatrix) traffic demands in matrix form
        :param allowed_degree: (nparray) a vector for degree constraints
        :param n_steps: (int) n_steps optimal is what we want
        :return: optimal final graph: (nxdict) 
        """
        self.n_nodes = n_nodes
        self.graph = nx.from_dict_of_dicts(graph)
        self.demand = demand
        self.allowed_degree = allowed_degree
        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        self.min_cost = self.cal_cost()
        cost = self.min_cost
        self.best_graph = graph
        curr_dict = []
        for _ in range(n_steps):
            curr_dict.append({})
        origin = nx.to_dict_of_lists(self.graph)
        notdone = True
        while(notdone):
            curr_dict, notdone = self.next_dict(origin, curr_dict, n_steps, (cost<self.inf))
            cost = self.multistep_predict(n_nodes, graph, curr_dict)
            if cost < self.min_cost:
                self.min_cost = cost
                self.best_graph = nx.to_dict_of_dicts(self.graph)
        return self.min_cost, self.best_graph
        
    def next_dict(self, origin, prev, n_steps, sign=False):
        """
        :param origin: (dict of lists) original graph
        :param prev: (list of dicts) [{add:[,],neigh:{left:,right:}},...]
        :param n_steps: (int)
        :return: dic:(dict) {add:[,],neigh:{left:,right:}}
        :return: success:(bool) whether successfully found the next dic
        """
        dic = prev.copy()
        for i in range(n_steps):
            ind = n_steps-1-i
            if "add" not in prev[ind]:
                prev_add = []
                for j in range(n_steps):
                    if "add" in prev[j]:
                        prev_add.append(prev[j]["add"])
                add_nodes, exist_add = self.next_to_add(self.n_nodes, n_steps, origin, prev_add, [0,0])
                if exist_add:
                    dic[ind]["add"] = add_nodes
                    for j in range(ind+1,n_steps):
                        dic[j].clear()
                    return dic, True
                else:
                    continue
            add_nodes = prev[ind]["add"]
            if "neigh" not in prev[ind]:
                if sign and i == 0:
                    prev_add = []
                    for j in range(n_steps):
                        if "add" in prev[j]:
                            prev_add.append(prev[j]["add"])
                    next_add_nodes, exist_add = self.next_to_add(self.n_nodes, n_steps, origin, prev_add, add_nodes)
                    if exist_add:
                        dic[ind]["add"] = next_add_nodes
                        for j in range(ind+1,n_steps):
                            dic[j].clear()
                        return dic, True
                    else:
                        continue
                else:
                    dic[ind]["neigh"] = {}
                    dic[ind]["neigh"]["right"] = origin[add_nodes[1]][0]
                    return dic, True
            else:
                if "right" not in prev[ind]["neigh"]:
                    dic[ind]["neigh"]["right"] = origin[add_nodes[1]][0]
                    for j in range(ind+1,n_steps):
                        dic[j].clear()
                    return dic, True
                elif prev[ind]["neigh"]["right"] != origin[add_nodes[1]][-1]:
                    current = origin[add_nodes[1]].index(prev[ind]["neigh"]["right"])
                    dic[ind]["neigh"]["right"] = origin[add_nodes[1]][current+1]
                    for j in range(ind+1,n_steps):
                        dic[j].clear()
                    return dic, True
                elif "left" not in prev[ind]["neigh"]:
                    dic[ind]["neigh"]["left"] = origin[add_nodes[0]][0]
                    del dic[ind]["neigh"]["right"]
                    for j in range(ind+1,n_steps):
                        dic[j].clear()
                    return dic, True
                elif prev[ind]["neigh"]["left"] != origin[add_nodes[0]][-1]:
                    current = origin[add_nodes[0]].index(prev[ind]["neigh"]["left"])
                    dic[ind]["neigh"]["left"] = origin[add_nodes[0]][current+1]
                    del dic[ind]["neigh"]["right"]
                    for j in range(ind+1,n_steps):
                        dic[j].clear()
                    return dic, True
                else:
                    prev_add = []
                    for j in range(n_steps):
                        if "add" in prev[j]:
                            prev_add.append(prev[j]["add"])
                    next_add_nodes, exist_next_add = self.next_to_add(self.n_nodes, n_steps, origin, prev_add, add_nodes)
                    if exist_next_add:
                        dic[ind]["add"] = next_add_nodes
                        del dic[ind]["neigh"]
                        for j in range(ind+1,n_steps):
                            dic[j].clear()
                        return dic, True
                    else:
                        continue
        return [], False

    def next_to_add(self, n_nodes, n_steps, graph, moves, current_move):
        for v1 in range(current_move[0], n_nodes):
            for v2 in range(v1+1,n_nodes):
                if v1 == current_move[0] and v2 <= current_move[1]:
                    continue
                if v2 in graph[v1]:
                    continue
                if [v1,v2] in moves:
                    continue
                return [v1,v2], True
        return [], False
    
    def multistep_predict(self, n_nodes, graph, moves):
        """
        :param n_nodes: (int)
        :param graph: (nxdict)
        :param moves: (list of dicts) each item is each step's move, {add:[v1,v2],neigh:{left: .., right: ..}}
        """
        valid = True
        self.graph = nx.from_dict_of_dicts(graph)
        for i in range(len(moves)):
            if "add" in moves[i]:
                add_nodes = moves[i]["add"]
                self._add_edge(add_nodes[0], add_nodes[1])
                if "neigh" in moves[i]:
                    if "left" in moves[i]["neigh"]:
                        if not self.graph.has_edge(add_nodes[0],moves[i]["neigh"]["left"]):
                            cost = self.inf
                            valid = False
                            break
                        self._remove_edge(add_nodes[0],moves[i]["neigh"]["left"])
                    if "right" in moves[i]["neigh"]:
                        if not self.graph.has_edge(add_nodes[1],moves[i]["neigh"]["right"]):
                            cost = self.inf
                            valid = False
                            break
                        self._remove_edge(add_nodes[1],moves[i]["neigh"]["right"])
                if not (nx.is_connected(self.graph) and self._check_degree(add_nodes[0]) and self._check_degree(add_nodes[1])):
                    cost = self.inf
                    valid = False
                    break
        if valid:
            cost = self.cal_cost()
        return cost
        
    def optimal_topology(self, n_nodes, demand, allowed_degree):
        """Searching for the best topology.
        :param n_nodes: (int) 
        :param demand: (nparray)
        :param allowed_degree: (nparray)
        """
        max_edges = int(n_nodes * (n_nodes-1) / 2)
        #min_edges = n_nodes - 1
        all_edges = list(range(max_edges))
        min_cost = self.inf
        best_graph = {}
        graph_dict = {}
        """
        for i in range(min_edges, max_edges + 1):
            if not i == 16:
                continue
            edges_comb = list(itertools.combinations(all_edges,i))
            cnt = 0
            for edges in edges_comb:
                graph_dict.clear()
                for j in range(n_nodes):
                    graph_dict[j] = []
                for e in edges:
                    [n1,n2] = edge_to_node(n_nodes, e)
                    graph_dict[n1].append(n2)
                    graph_dict[n2].append(n1)
                cost = self.cal_cost_judge(n_nodes,graph_dict,demand,allowed_degree)
                if cost < min_cost:
                    min_cost = cost
                    best_graph = graph_dict
                cnt += 1
                if cnt % 10000 == 0:
                    print("checked: {}".format(cnt))
        """
        edges_comb = list(itertools.combinations(all_edges,n_nodes*2))
        #cnt = 0
        for edges in edges_comb:
            graph_dict.clear()
            for j in range(n_nodes):
                graph_dict[j] = []
            for e in edges:
                [n1,n2] = edge_to_node(n_nodes, e)
                graph_dict[n1].append(n2)
                graph_dict[n2].append(n1)
            cost = self.cal_cost_judge(n_nodes,graph_dict,demand,allowed_degree)
            if cost < min_cost:
                min_cost = cost
                best_graph = graph_dict
            #cnt += 1
            #if cnt % 500000 == 0:
            #    print("checked: {}".format(cnt))

        return min_cost, best_graph

    def optimal_topology_mp(self, n_nodes, demand, degree):
        print("patching data..")
        max_edges = int(n_nodes * (n_nodes-1) / 2)
        all_edges = list(range(max_edges))
        edges_comb = list(itertools.combinations(all_edges,n_nodes*degree/2))
        params = []
        for edges in edges_comb:
            param = {}
            param["edges"] = edges
            param["n"] = n_nodes
            param["demand"] = demand
            param["degree"] = degree
            params.append(param)
        print("Finished patching")
        pool = Pool()
        costs = pool.map(self.test_topo_run, params)
        pool.close()
        pool.join()
        min_cost = np.min(np.array(costs))
        return min_cost

    def test_topo_run(self, param):
        n_nodes = param['n']
        edges = param['edges']
        demand = param['demand']
        allowed_degree = param['degree'] * np.ones((n_nodes,))
        graph_dict = {}
        for j in range(n_nodes):
            graph_dict[j] = []
        for e in edges:
            [n1,n2] = edge_to_node(n_nodes, e)
            graph_dict[n1].append(n2)
            graph_dict[n2].append(n1)
        cost = self.cal_cost_judge(n_nodes,graph_dict,demand,allowed_degree)
        return cost

    def multistep_BFS(self,n_nodes,graph,demand,allowed_degree,n_steps=1):
        """
        Args:
            n_nodes: (int) the total number of nodes in the network
            graph: (nxdict) current topology
            demand: (npmatrix) traffic demands in matrix form
            allowed_degree: (nparray) a vector for degree constraints
            n_steps: (int) n_steps optimal is what we want
        Returns:
            optimal final graph: (nxdict) 
        """
        self.n_nodes = n_nodes
        self.graph = nx.from_dict_of_dicts(graph)
        self.demand = demand
        self.allowed_degree = allowed_degree
        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse
        
        toposet = [graph]
        for _ in range(n_steps):
            self.one_more_step(n_nodes, toposet)
        costs = []
        for topo in toposet:
            self.graph = nx.from_dict_of_dicts(topo)
            cost = self.cal_cost()
            costs.append(cost)
        ind = costs.index(min(costs))
        return toposet[ind]

    def consturct_v(self, best_action, neighbor_to_remove):
        """Constructiong a vector V for supervised learning.
        Args: 
            best_action: (list) the node pair to be added
            neighbor_to_remove: (dict) keys may contains "left" and "right"
        
        Returns:
            v: (nparray) a vector V for voltages
        """
        v = np.zeros((self.n_nodes,),np.float32)
        if len(best_action) == 0:
            return v
        upper_demand = np.triu(self.demand,k=1)
        lower_demand = np.tril(self.demand,k=-1)        
        sum_demand_u = np.sum(upper_demand)
        sum_demand_l = np.sum(lower_demand)
        v1 = best_action[0]
        v2 = best_action[1]
        v[v1] = -1.0
        v[v2] = 1.0
        if "left" in neighbor_to_remove:
            v[neighbor_to_remove["left"]] = -0.5
        if "right" in neighbor_to_remove:
            v[neighbor_to_remove["right"]] = 0.5
        if sum_demand_l > sum_demand_u:
            v = -v
        return v

    def consideration(self,v1,v2):
        neighbors1 = [n for n in self.graph.neighbors(v1)]
        neighbors2 = [n for n in self.graph.neighbors(v2)]
        self._add_edge(v1,v2)
        costs = []
        if not self._check_degree(v1) and not self._check_degree(v2): 
            for left in range(len(neighbors1)):
                for right in range(len(neighbors2)):
                    n_l = neighbors1[left]
                    n_r = neighbors2[right]
                    self._remove_edge(n_l,v1)
                    self._remove_edge(n_r,v2)
                    if not nx.is_connected(self.graph):
                        costs.append(self.inf)
                    else:
                        cost = self.cal_cost()
                        costs.append(cost)
                    self._add_edge(n_l,v1)
                    self._add_edge(n_r,v2)
            ind = costs.index(min(costs))
            ind_left = ind // len(neighbors2)
            ind_right = ind % len(neighbors2)
            neigh = {"left":neighbors1[ind_left],"right":neighbors2[ind_right]}
        elif not self._check_degree(v1):
            for left in range(len(neighbors1)):
                n_l = neighbors1[left]
                self._remove_edge(n_l,v1)
                if not nx.is_connected(self.graph):
                    costs.append(self.inf)
                else:
                    cost = self.cal_cost()
                    costs.append(cost)
                self._add_edge(n_l,v1)
            ind_left = costs.index(min(costs))
            neigh = {"left":neighbors1[ind_left]}
        elif not self._check_degree(v2):
            for right in range(len(neighbors2)):
                n_r = neighbors2[right]
                self._remove_edge(n_r,v2)
                if not nx.is_connected(self.graph):
                    costs.append(self.inf)
                else:
                    cost = self.cal_cost()
                    costs.append(cost)
                self._add_edge(n_r,v2)
            ind_right = costs.index(min(costs))
            neigh = {"right":neighbors2[ind_right]}
        else:
            cost = self.cal_cost()
            costs.append(cost)
            neigh = dict()
        min_cost = min(costs)
        self._remove_edge(v1,v2)
        return min_cost, neigh


    def one_more_step(self, n_nodes, topos):
        """
        :param n_nodes: (int) the number of nodes in the network
        :param topos: (list) a list of nxdicts, last step topologies
        :return: new_graphs: (list) a list of nxdicts
        """
        graphs = copy.deepcopy(topos)
        for topo in graphs:
            self.graph = nx.from_dict_of_dicts(topo)
            #unchanged = False
            for v1 in range(n_nodes-1):
                for v2 in range(v1+1,n_nodes):
                    if self.graph.has_edge(v1,v2):
                        continue
                    neighbors1 = [n for n in self.graph.neighbors(v1)]
                    neighbors2 = [n for n in self.graph.neighbors(v2)]
                    self._add_edge(v1,v2)
                    if not self._check_degree(v1) and not self._check_degree(v2): 
                        for left in range(len(neighbors1)):
                            for right in range(len(neighbors2)):
                                n_l = neighbors1[left]
                                n_r = neighbors2[right]
                                self._remove_edge(n_l,v1)
                                self._remove_edge(n_r,v2)
                                if nx.is_connected(self.graph):
                                    new_topo = nx.to_dict_of_dicts(self.graph)
                                    topos.append(new_topo)
                                #else:
                                #    unchanged = True
                                self._add_edge(n_l,v1)
                                self._add_edge(n_r,v2)
                    elif not self._check_degree(v1):
                        for left in range(len(neighbors1)):
                            n_l = neighbors1[left]
                            self._remove_edge(n_l,v1)
                            if nx.is_connected(self.graph):
                                new_topo = nx.to_dict_of_dicts(self.graph)
                                topos.append(new_topo)
                            #else:
                            #    unchanged = True
                            self._add_edge(n_l,v1)
                    elif not self._check_degree(v2):
                        for right in range(len(neighbors2)):
                            n_r = neighbors2[right]
                            self._remove_edge(n_r,v2)
                            if nx.is_connected(self.graph):
                                new_topo = nx.to_dict_of_dicts(self.graph)
                                topos.append(new_topo)
                            #else:
                            #    unchanged = True
                            self._add_edge(n_r,v2)
                    else:
                        new_topo = nx.to_dict_of_dicts(self.graph)
                        topos.append(new_topo)

                    self._remove_edge(v1,v2)
        #return new_graphs
    
    def cal_cost(self):
        cost = 0
        for s, d in itertools.product(range(self.n_nodes), range(self.n_nodes)):
            try:
                path_length = float(nx.shortest_path_length(self.graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                path_length = float(self.n_nodes)

            cost += path_length * self.demand[s][d]
        
        return cost

    def cal_cost_judge(self, n_nodes, graph_dict, demand, degree):
        for i in range(n_nodes):
            if len(graph_dict[i]) > degree[i]:
                return self.inf

        graph = nx.from_dict_of_lists(graph_dict)
        if not nx.is_connected(graph):
            return self.inf

        cost = 0
        for s, d in itertools.product(range(n_nodes), range(n_nodes)):
            try:
                path_length = float(nx.shortest_path_length(graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                path_length = float(n_nodes)
            cost += path_length * demand[s][d]
        return cost

    def _add_edge(self, v1, v2):
        self.graph.add_edge(v1,v2)

        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse

    def _remove_edge(self, v1, v2):
        self.graph.remove_edge(v1, v2)
        # Update available degree
        degree_inuse = np.array(self.graph.degree)[:,-1]
        self.available_degree = self.allowed_degree - degree_inuse

    def _check_degree(self, node):
        if self.available_degree[node] < 0:
            return False
        else:
            return True

def edge_to_node(node_num,e):
    c = 0
    for i in range(node_num-1):
        for j in range(i+1,node_num):
            if c == e:
            #if ((i*(2*node_num-1-i)/2-1+j-i)== e):
                return [i,j]
            c += 1

"""
opt = optimal()
topo = {0: {7: {}}, 1: {4: {}, 7: {}}, 2: {4: {}, 5: {}, 6: {}, 7: {}}, 3: {4: {}, 5: {}, 6: {}}, 4: {1: {}, 2: {}, 3: {}, 6: {}}, 5: {2: {}, 3: {}, 6: {}, 7: {}}, 6: {2: {}, 3: {}, 4: {}, 5: {}}, 7: {0: {}, 1: {}, 2: {}, 5: {}}}
demand = np.array([
[0, 4, 1, 1, 3, 4, 5, 1],
[4, 0, 4, 3, 5, 4, 2, 1],
[3, 2, 0, 3, 4, 2, 3, 1],
[0, 5, 2, 0, 2, 7, 3, 2],
[1, 4, 4, 3, 0, 1, 4, 2],
[5, 2, 1, 3, 1, 0, 4, 5],
[4, 6, 1, 1, 3, 0, 0, 3],
[4, 2, 3, 7, 3, 3, 3, 0]
])

degree = np.array([4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

optdict = opt.multistep_compute_optimal(8,topo,demand,degree)
"""