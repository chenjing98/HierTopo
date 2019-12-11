import networkx as nx
import tensorflow as tf
import models
from layers import GraphConvolution
import numpy as np
import TopoEnv 
import networkx as nx
import matplotlib.pyplot as plt
import edge_graph
import utils
import multiprocessing as mp
import pickle as pk
import time
from agent import ActorAgent
from tf_logger import TFLogger
from param import args
from average_reward import *

S_LEN = 7
A_DIM = 3
NUM_AGENTS = 1
NUM_NODES = 8
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100


def main():
    pass


def invoke_model(actor_agent, edges, exp):
    edge_acts, edge_act_probs, stop = actor_agent.invoke_model(edges)
    s, d = edge_graph.cal_node_id(edge_acts, args.num_nodes)
    action = [s, d, round(stop)]
    exp['action'].append(action)
    return action


def train_agent(agent_id, param_queue, reward_queue, adv_queue, gradient_queue):
    tf.set_random_seed(agent_id)
    env = TopoEnv.TopoEnv()
    config = tf.ConfigProto(device_count={'GPU': args.worker_num_cpu}, 
                gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=worker_gpu_fraction))
    sess = tf.Session(config=config)

    assert env.edge_graph.number_of_nodes == args.num_nodes * (args.num_nodes - 1) / 2
    actor_agent = ActorAgent(sess=sess, node_input_dim=env.edge_graph.number_of_nodes, 
                             hid_dims=args.hid_dims, output_dim=args.output_dim,
                             max_depth=args.max_depth)

    while True:
        (actor_params, seed, entropy_weight) = param_queue.get()
        actor_agent.set_params(actor_params)

        env.seed(seed)
        env.reset()

        edges = env.edge_graph
        done = False

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


        edge_nodes = (num_nodes - 1) * num_nodes / 2 
        actor = models.ActorNetwork(sess, feature_dim=[edge_nodes, S_LEN], 
                                    action_dim=[edge_nodes, A_DIM], learning_rate=ACTOR_LR_RATE)
        critic = models.CriticNetwork(sess, feature_dim=[edge_nodes, S_LEN], 
                                      learning_rate=CRITIC_LR_RATE)

        summary_ops, summary_vars = models.build_summaries()
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./', sess.graph)
        saver = tf.train.Saver()

        epoch = 0
        action = np.append(np.unravel_index(np.argmax(demand), demand.shape), 0)

        adj_batch = [np.zeros((num_nodes, num_nodes))]
        f_batch = [np.zeros((num_nodes, S_LEN))]
        a_batch = [action]
        r_batch = []
        entropy_record = []
        actor_gradient_batch = []
        critic_gradient_batch = []

        edges = env.edge_graph
        while True:

            adj = np.array(nx.adjacency_matrix(edges).todense())
            features = edge_graph.get_features(edges)
            adj_batch.append(adj)
            f_batch.append(features)
            r_batch.append(reward)

            # ATTENTION !!! edegs.number_of_nodes != num_nodes !!!
            action = actor.predict(np.reshape(adj, (1, edges.number_of_nodes, edges.number_of_nodes)), 
                                   np.reshape(features, (1, edges.number_of_nodes, S_LEN)))

            edges, reward, stop = env.step(action)

            
            if len(r_batch) > TRAIN_SEQ_LEN or stop:
                actor_gradient, critic_gradient, td_batch = models.compute_gradients(
                    adj_batch=np.stack(adj_batch[1:], axis=0),
                    f_batch=np.stack(f_batch[1:], axis=0),
                    a_batch=np.stack(a_batch[1:], axis=0),
                    r_batch=np.vstack(r_batch[1:]),
                    terminal=stop, actor=actor, critic=critic
                )
                td_loss = np.mean(td_batch)

                actor_gradient_batch.append(actor_gradient_batch)
                critic_gradient_batch.append(critic_gradient)

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: td_loss,
                    summary_vars[1]: np.mean(r_batch),
                    summary_vars[2]: np.mean(entropy_record)
                })

                writer.add_summary(summary_str, epoch)
                writer.flush()
                entropy_record = []

                if len(actor_gradient_batch) >= GRADIENT_BATCH_SIZE:
                    for i in range(len(actor_gradient_batch)):
                        actor.apply_gradients(actor_gradient_batch[i])
                        critic.apply_gradients(critic_gradient_batch[i])

                    actor_gradient_batch = []
                    critic_gradient_batch = []

                    epoch += 1
                    if epoch % MODEL_SAVE_INTERVAL == 0:
                        print(saver.save(sess, './nn_model_ep_' + str(epoch) + '.ckpt'))        

                del adj_batch[:]
                del f_batch[:]
                del a_batch[:]
                del r_batch[:]

            adj_batch.append(adj)
            f_batch.append(f)
            a_batch.append(action)


if __name__ == "__main__":
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # create result and model folder
    create_folder_if_not_exists(args.result_folder)
    create_folder_if_not_exists(args.model_folder)

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

    actor_agent = ActorAgent(sess=sess, node_input_dim=env.edge_graph.number_of_nodes, 
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

        actor_agent.apply_gradients(
            aggregate_gradients(actor_gradients), args.lr)

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
