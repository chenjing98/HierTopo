import numpy as np
import tensorflow as tf
import utils
import tensorflow.contrib.layers as tl
import networkx as nx

import edge_graph
from gcn import GraphCNN
from param import args


# super class of scheduling agnet
class Agent(object):
    def __init__(self):
        pass

    def get_action(self, obs):
        print('get_action not implemented')
        exit(1)


class ActorAgent(Agent):
    def __init__(self, sess, node_input_dim, hid_dims, output_dim, max_depth, eps=1e-6, 
                 act_fn=tf.nn.leaky_relu, optimizer=tf.train.AdamOptimizer, scope='actor_agent'):

        Agent.__init__(self)

        self.sess = sess
        self.node_input_dim = node_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope

        # node_input_dim = 7
        self.node_inputs = tf.placeholder(tf.float32, shape=[None, self.node_input_dim])

        self.gcn = GraphCNN(inputs=self.node_inputs, input_dim=self.node_input_dim,
                            hid_dims=self.hid_dims, output_dim=self.output_dim, 
                            max_depth=self.max_depth, act_fn=self.act_fn, scope=self.scope)
        
        # map gcn_outputs and raw_inputs to action probabilities
        # edge_act_probs: [batch_size, total_edge_nodes -> n(n-1)/2]
        # stop: results after sigmoid. stop (~1) or non-stop (~0)
        self.edge_act_probs, self.stop = self.actor_network(
            node_inputs=self.node_inputs, 
            gcn_outputs=self.gcn.outputs, 
            act_fn=self.act_fn)
        self.edge_acts = tf.argmax(self.edge_act_probs, 1)
        
        # advantage term (from Monte Calro or critic) ([batch_size, 1])
        self.adv = tf.placeholder(tf.float32, [None, 1])

        # use entropy to promote exploration, this term decays over time
        self.entropy_weight = tf.placeholder(tf.float32, ())

        # Selected action for edge, 0-1 vector ([batch_size, total_edge_nodes -> n(n-1)/2])
        self.edge_act_vec = tf.placeholder(tf.float32, [None, None])

        # select edge action probability
        self.selected_edge_prob = tf.reduce_sum(tf.multiply(self.edge_act_probs, self.edge_act_vec),
                                                reduction_indices=1, keep_dims=True)

        # actor loss due to advantge (negated)
        self.adv_loss = tf.reduce_sum(tf.multiply(tf.log(self.selected_edge_prob + self.eps),
                                                  -self.adv))

        # edge_entropy
        self.edge_entropy = tf.reduce_sum(tf.multiply(self.edge_act_probs, 
                                                      tf.log(self.edge_act_probs + self.eps)))
        # entropy loss
        self.entropy_loss = self.edge_entropy

        # normalize entropy
        self.entropy_loss /= tf.log(tf.cast(tf.shape(self.edge_act_probs)[1], tf.float32))

        # define combined loss
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss

        # get training parameters
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)

        # operations for setting network parameters
        self.input_params, self.set_params_op = self.define_params_op()

        # actor gradients
        self.act_gradients = tf.gradients(self.act_loss, self.params)

        # adaptive learning rate
        self.lr_rate = tf.placeholder(tf.float32, shape=[])

        # actor optimizer
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        # apply gradient directly to update parameters
        self.apply_grads = self.optimizer(self.lr_rate).\
            apply_gradients(zip(self.act_gradients, self.params))

        # network paramter saver
        self.saver = tf.train.Saver(max_to_keep=args.num_saved_models)
        self.sess.run(tf.global_variables_initializer())

        if args.saved_model is not None:
            self.saver.restore(self.sess, args.saved_model)

    def actor_network(self, node_inputs, gcn_outputs, act_fn):

        # takes output from graph embedding and raw_input from environment

        batch_size = tf.shape(node_inputs)[0]

        # (1) reshape node inputs to batch format
        node_inputs_reshape = tf.reshape(node_inputs, [batch_size, -1, self.node_input_dim])

        # (4) reshape gcn_outputs to batch format
        gcn_outputs_reshape = tf.reshape(gcn_outputs, [batch_size, -1, self.output_dim])

        # (4) actor neural network
        with tf.variable_scope(self.scope):
            # -- part A, the distribution over nodes --
            merge_node = tf.concat([node_inputs_reshape, gcn_outputs_reshape], axis=2)

            node_hid_0 = tl.fully_connected(merge_node, 32, activation_fn=act_fn)
            node_hid_1 = tl.fully_connected(node_hid_0, 16, activation_fn=act_fn)
            node_hid_2 = tl.fully_connected(node_hid_1, 8, activation_fn=act_fn)
            node_outputs = tl.fully_connected(node_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, total_edge_nodes)
            node_outputs = tf.reshape(node_outputs, [batch_size, -1])

            # do softmax over nodes on the graph
            node_outputs = tf.nn.softmax(node_outputs, dim=-1)

            # -- part B, the distribution over stop --
            stop_hid_0 = tl.fully_connected(merge_node, 32, activation_fn=act_fn)
            stop_hid_1 = tl.fully_connected(stop_hid_0, 16, activation_fn=act_fn)
            stop_hid_2 = tl.fully_connected(stop_hid_1, 8, activation_fn=act_fn)
            stop_outputs = tl.fully_connected(stop_hid_2, 1, activation_fn=None)

            # reshape the output dimension (batch_size, num_jobs * num_exec_limits)
            stop_outputs = tf.sigmoid(stop_outputs)

            return node_outputs, stop_outputs

    def apply_gradients(self, gradients, lr_rate):
        self.sess.run(self.apply_grads, feed_dict={
            i: d for i, d in zip(self.act_gradients + [self.lr_rate], gradients + [lr_rate])
        })

    def define_params_op(self):
        # define operations for setting network parameters
        input_params = []
        for param in self.params:
            input_params.append(tf.placeholder(tf.float32, shape=param.get_shape()))
        set_params_op = []
        for idx, param in enumerate(input_params):
            set_params_op.append(self.params[idx].assign(param))
        return input_params, set_params_op
    
    # node and edge here are referred to the same.
    def predict(self, node_inputs, gcn_mats):
        print(np.shape(node_inputs), np.shape(gcn_mats))
        print(np.shape(self.node_inputs), np.shape(self.gcn.adj_mats))
        return self.sess.run([self.edge_act_probs, self.edge_acts, self.stop], feed_dict={
            i: d for i, d in zip([self.node_inputs] + [self.gcn.adj_mats], [node_inputs] + [gcn_mats])
        })
    
    def invoke_model(self, edges):
        # implement this module here for training
        # (to pick up state and action to record)

        # invoke learning model
        node_inputs = edge_graph.get_features(edges)
        gcn_mats = nx.adjacency_matrix(edges).todense()
        edge_act_probs, edge_acts, stop = self.predict(node_inputs, gcn_mats)

        return edge_acts, edge_act_probs, stop

    def get_params(self):
        return self.sess.run(self.params)

    def set_params(self, input_params):
        self.sess.run(self.set_params_op, feed_dict={
            i: d for i, d in zip(self.input_params, input_params)
        })

    