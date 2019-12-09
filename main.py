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

S_LEN = 7
A_DIM = 3
NUM_AGENTS = 1
NUM_NODES = 8
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100


def main():

    num_nodes = NUM_NODES
    num_ep = 10000
    with open('10M_8_3.0_const3.pk3', 'rb') as f:
        dataset = pk.load(f)
    env = TopoEnv.TopoEnv(dataset=dataset)

    with tf.Session() as sess:
        actor = models.ActorNetwork(sess, feature_dim=[num_nodes, S_LEN], action_dim=A_DIM, 
                                    learning_rate=ACTOR_LR_RATE)
        critic = models.CriticNetwork(sess, feature_dim=[num_nodes, S_LEN], 
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


        while True:
            edges, reward, stop = env.step(action)

            adj = np.array(nx.adjacency_matrix(edges).todense())
            features = edge_graph.get_features(edges)
            adj_batch.append(adj)
            f_batch.append(features)
            r_batch.append(reward)

            action = actor.predict(np.reshape(adj, (1, num_nodes, num_nodes)), 
                                   np.reshape(features, (1, num_nodes, S_LEN)))
            
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
    main()
