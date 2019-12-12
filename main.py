import networkx as nx
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import multiprocessing as mp
import pickle as pk
import time

from layers import GraphConvolution
import numpy as np
import models
import TopoEnv 
import edge_graph
import utils
from agent import ActorAgent
from tf_logger import TFLogger
from param import args
from average_reward import *

S_LEN = 7
A_DIM = 3
NUM_AGENTS = 1
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100


def invoke_model(actor_agent, edges, exp):
    edge_acts, edge_act_probs, stop = actor_agent.invoke_model(edges)
    s, d = edge_graph.cal_node_id(edge_acts, args.num_nodes)
    action = [s, d, round(stop)]
    exp['action'].append(action)
    return action


def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    tf.set_random_seed(agent_id)
    env = TopoEnv.TopoEnv()
    config = tf.ConfigProto(device_count={'GPU': args.worker_num_gpu}, 
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.worker_gpu_fraction))
    sess = tf.Session(config=config)

    assert int(env.edges.number_of_nodes()) == int(args.num_nodes * (args.num_nodes - 1) / 2)
    actor_agent = ActorAgent(sess=sess, node_input_dim=args.node_input_dim, 
                             hid_dims=args.hid_dims, output_dim=args.output_dim,
                             max_depth=args.max_depth)

    while True:
        (actor_params, seed, entropy_weight) = param_queue.get()
        actor_agent.set_params(actor_params)

        env.seed(seed)

        edges = env.edges
        stop = False

        exp = {'edges': [], 'reward': []}

        while not stop:
            action = invoke_model(actor_agent, edges, exp)
            edges, reward, stop = env.step(action)
            exp['edges'].append(edges)
            exp['reward'].append(reward)
        
        reward_queue.put([exp['reward']])
        batch_adv = adv_queue.get()
        if batch_adv is None:
            continue
        actor_gradient, loss = compute_actor_gradients(actor_agent, exp, batch_adv, entropy_weight)
        gradient_queue.put([actor_gradient, loss])


if __name__ == "__main__":
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    # initialize communication queues
    params_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    reward_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    adv_queues = [mp.Queue(1) for _ in range(args.num_agents)]
    gradient_queues = [mp.Queue(1) for _ in range(args.num_agents)]

    # set up training agents
    agents = []
    for i in range(args.num_agents):
        agents.append(mp.Process(target=train_agent, args=(
            i, params_queues[i], reward_queues[i],
            adv_queues[i], gradient_queues[i])))

    # start training agents
    for i in range(args.num_agents):
        agents[i].start()

    # gpu configuration
    config = tf.ConfigProto(device_count={'GPU': args.master_num_gpu},
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=args.master_gpu_fraction))

    sess = tf.Session()

    actor_agent = ActorAgent(sess=sess, node_input_dim=args.node_input_dim, 
                             hid_dims=args.hid_dims, output_dim=args.output_dim,
                             max_depth=args.max_depth)
    tf_logger = TFLogger(sess, ['actor_loss', 'entropy',  'value_loss', 'episode_length',
        'sum_reward', 'entropy_weight'])

    avg_reward_calculator = AveragePerStepReward(args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    # reset_prob = args.reset_prob

    # ---- start training process ----
    for ep in range(1, args.num_ep):
        print('training epoch', ep)

        # synchronize the model parameters for each training agent
        actor_params = actor_agent.get_params()

        # send out parameters to training agents
        for i in range(args.num_agents):
            params_queues[i].put([actor_params, args.seed + ep, entropy_weight])
        
        all_rewards = []

        t1 = time.time()

        for i in range(args.num_agents):
            result = reward_queues[i].get()
            batch_reward = result
            all_rewards.append(batch_reward)
            avg_reward_calculator.add_list(batch_reward)

        t2 = time.time()
        print('got reward from workers', t2 - t1, 'seconds')

        # compute differential reward
        all_cum_reward = []
        # avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(args.num_agents):
            # regular reward
            rewards = np.array(all_rewards)
            cum_reward = discount(rewards, args.gamma)
            all_cum_reward.append(cum_reward)

        # compute baseline (might be removed? specific for input-driven env?)
        baselines = utils.get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # give worker back the advantage
        for i in range(args.num_agents):
            batch_adv = all_cum_reward[i] - baselines[i]
            batch_adv = np.reshape(batch_adv, [len(batch_adv), 1])
            adv_queues[i].put(batch_adv)

        t3 = time.time()
        print('advantage ready', t3 - t2, 'seconds')

        actor_gradients = []
        all_action_loss, all_entropy, all_value_loss = [], [], []   # for tensorboard

        for i in range(args.num_agents):
            (actor_gradient, loss) = gradient_queues[i].get()

            actor_gradients.append(actor_gradient)
            all_action_loss.append(loss[0])
            all_entropy.append(-loss[1] / float(all_cum_reward[i].shape[0]))
            all_value_loss.append(loss[2])

        t4 = time.time()
        print('worker send back gradients', t4 - t3, 'seconds')

        actor_agent.apply_gradients(aggregate_gradients(actor_gradients), args.lr)

        t5 = time.time()
        print('apply gradient', t5 - t4, 'seconds')

        tf_logger.log(ep, [
            np.mean(all_action_loss),
            np.mean(all_entropy),
            np.mean(all_value_loss),
            np.mean([len(b) for b in baselines]),
            np.mean([cr[0] for cr in all_cum_reward]),
            entropy_weight])

        if ep % args.model_save_interval == 0:
            actor_agent.save_model(args.model_folder + \
                'model_ep_' + str(ep))

    sess.close()
