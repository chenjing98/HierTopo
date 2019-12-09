import numpy as np
import tensorflow as tf
import gcn


class ActorAgent(Agent):
    def __init__(self, sess, node_input_dim, job_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, eps=1e-6, act_fn=leaky_relu,
                 optimizer=tf.train.AdamOptimizer, scope='actor_agent'):

        Agent.__init__(self)

        self.sess = sess
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope

        