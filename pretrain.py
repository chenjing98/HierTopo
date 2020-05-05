#import os
#import gym
import numpy as np
import itertools
import networkx as nx
import pickle as pk

#from stable_baselines import PPO2

#from topoenv import TopoEnv
from whatisoptimal import optimal
from baseline_new.permatch import permatch

class pretrainDataset(object):
    def __init__(self,num_n=8,
                      fpath='10M_8_3.0_const3_pretrain.pk3',
                      fpath_topo='10M_8_3.0_const3_topo_pretrain.pk3'):

        self.num_n = num_n
        self.opt = optimal()
        self.permatch_baseline = permatch(num_n)
        with open(fpath, 'rb') as f1:
            self.dataset = pk.load(f1)
        with open(fpath_topo, 'rb') as f2:
            self.topo_dataset = pk.load(f2)
    
    def collect_data(self, dataset_size=64000):
        """
        The structure of the expert dataset is a dict, saved as an “.npz” archive. 
        The dictionary contains the keys ‘actions’, ‘episode_returns’, ‘rewards’, ‘obs’ and ‘episode_starts’. 
        The corresponding values have data concatenated across episode: 
            the first axis is the timestep, the remaining axes index into the data.
        """
        actions = []
        episode_returns = []
        obs = []
        episode_starts = []

        for idx in range(dataset_size):
            demand = self.dataset[idx]['demand']
            allowed_degree = self.dataset[idx]['allowed_degree']
            topo_dict = self.topo_dataset[idx]
            topo = nx.from_dict_of_dicts(topo_dict)
            degree_inuse = np.array(topo.degree)[:,-1]
            available_degree = allowed_degree - degree_inuse
            adj = np.array(nx.adjacency_matrix(topo).todense(), np.float32)
            expand_availdeg = available_degree[np.newaxis,:]
            demand_norm = demand/(np.max(demand)+1e-7)
            ob = np.concatenate((demand_norm,adj,expand_availdeg),axis=0)
            ob = np.reshape(ob,((2*self.num_n+1)*self.num_n,))
            obs.append(ob)
            episode_starts.append(True)
            best_action, neigh, topo_new = self.opt.compute_optimal(self.num_n,topo,demand,allowed_degree)
            action = self.opt.consturct_v(best_action,neigh)
            actions.append(action)
            epi_return = self._cal_reward_against_permatch(demand,allowed_degree,topo,topo_new)
            episode_returns.append(epi_return)
            print("[Index {}]".format(idx))

        np.savez("./pretraindata.npz",
                actions=actions,
                episode_returns=episode_returns,
                obs=obs,
                episode_starts=episode_starts)

    def _cal_reward_against_permatch(self,demand,allowed_degree,topo,topo_new):
        nn_score = 0
        permatch_score = 0
        permatch_new_graph = self.permatch_baseline.n_steps_matching(
            demand,topo,allowed_degree,1) # weighted matching

        for s, d in itertools.product(range(self.num_n), range(self.num_n)):
            try:
                permatch_path_length = float(nx.shortest_path_length(permatch_new_graph,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                permatch_path_length = float(self.num_n)

            try:
                nn_path_length = float(nx.shortest_path_length(topo_new,source=s,target=d))
            except nx.exception.NetworkXNoPath:
                nn_path_length = float(self.num_n)

            permatch_score += permatch_path_length * demand[s][d]
            nn_score += nn_path_length * demand[s][d]
        
        #permatch_score /= (sum(sum(self.demand)))
        #nn_score /= (sum(sum(self.demand)))
        #last_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        #cur_score /= (sum(sum(self.demand)) * math.sqrt(self.max_node))
        return (permatch_score - nn_score)/permatch_score

def main():
    pretrain_dataset = pretrainDataset()
    pretrain_dataset.collect_data()


main()