import gym
import warnings
from itertools import zip_longest

import numpy as np
import tensorflow as tf
from gym.spaces import Discrete

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.distributions import make_proba_dist_type, CategoricalProbabilityDistribution, \
    MultiCategoricalProbabilityDistribution, DiagGaussianProbabilityDistribution, BernoulliProbabilityDistribution
from stable_baselines.common.input import observation_input

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.vec_env import DummyVecEnv

class GnnPolicy(ActorCriticPolicy):
    """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) (deprecated, use net_arch instead) The size of the Neural network for the policy
        (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture (see mlp_extractor
        documentation for details).
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None, net_arch=None,
                 act_fun=tf.tanh, scale=False, **kwargs):
        super(GnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                                scale=scale)

        # hyperparameters for GNN
        self.beta_v = 0.5
        self.beta_i = 0.5
        self.depths = 2

        self.input_dim = 1 # must be
        self.hid_dims = [64,64]
        self.out_dim = 64

        self.num_n = 8 # max_node in the environment

        if layers is not None:
            warnings.warn("Usage of the `layers` parameter is deprecated! Use net_arch instead "
                          "(it has a different semantics though).", DeprecationWarning)
            if net_arch is not None:
                warnings.warn("The new `net_arch` parameter overrides the deprecated `layers` parameter!",
                              DeprecationWarning)

        if net_arch is None:
            if layers is None:
                layers = [64, 64]
            net_arch = [dict(vf=layers, pi=layers)]

        with tf.variable_scope("gnn", reuse=reuse):
            
            self._obs_process(self.processed_obs)
            initial = True
            batch_size = tf.shape(self.processed_obs)[0]
            V0 = tf.zeros([self.num_n,self.num_n,self.num_n],dtype=tf.float32)
            I0 = tf.zeros([self.num_n,self.num_n,self.num_n,self.num_n],dtype=tf.float32)
            v_curr = V0
            i_curr = I0
            for i in range(self.depths):
                v_tplus1 = self.message_passing_V(v_curr,i_curr,self.adj,reuse_graph_tensor=initial)
                i_tplus1 = self.message_passing_I(i_curr,self.adj,self.demand,reuse_graph_tensor=initial)
                if initial:
                    initial = False
                v_curr = v_tplus1
                i_curr = i_tplus1
            graph_latent = v_curr
            #self.gnn_weights, self.gnn_bias = self._para_init()

            #graph_latent = self._egnn(self.processed_obs)

        with tf.variable_scope("model", reuse=reuse):

            pi_latent, vf_latent = mlp_extractor(tf.layers.flatten(graph_latent), net_arch, act_fun)

            self._value_fn = linear(vf_latent, 'vf', 1)

            self._proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
    
    def _obs_process(self, obs):
        self.demand = obs[:,:,:-1,:,:]
        self.adj = obs[:,0,-1,:,:]

    def message_passing_V(self, V_t, I_t, adj, reuse_graph_tensor=False):
        """Message passing & update for voltages
        Args:
            V_t: voltages at timestep t, of size batch_size x N x N x N
            I_t: currents at timestep t, of size batch_size x N x N x N x N
            adj: adjacent matrix, of size batch_size x N x N
            reuse_graph_tensor: a boolean for if it is the 1st time calling this function
        Returns:
            V_t+1: voltages at timestep t+1, of size batch_size x N x N x N
        """
        batch_size = tf.shape(adj)[0]
        if reuse_graph_tensor:
            self.R = tf.get_variable(name='r',
                    shape=[self.num_n,self.num_n,self.num_n,self.num_n],
                    dtype=tf.float32)
            self.w_v = tf.get_variable(name='w_v',
                    shape=[self.num_n,self.num_n],
                    dtype=tf.float32)
            self.b_v = tf.get_variable(name='b_v',
                    shape=[self.num_n],
                    dtype=tf.float32)
            V_t = tf.tile(tf.expand_dims(V_t,0),[batch_size,1,1,1])
            I_t = tf.tile(tf.expand_dims(I_t,0), [batch_size,1,1,1,1])
        delta_v = tf.multiply(self.R, I_t)
        m_temp = tf.matmul(tf.reshape(tf.transpose(I_t, [0,1,2,4,3]),[batch_size,self.num_n**3,self.num_n]), adj, transpose_b=True)
        m_temp = tf.transpose(tf.reshape(m_temp,[batch_size,self.num_n,self.num_n,self.num_n,self.num_n]), [0,1,2,4,3])
        m_origin = tf.reduce_sum(m_temp,-1) # of size [batch_size, N, N, N]
        m_v_tplus1 = tf.tanh(tf.matmul(
            tf.reshape(m_origin, [batch_size,self.num_n**2,self.num_n]),
            tf.tile(tf.expand_dims(self.w_v,0), [batch_size,1,1]))
            + tf.tile(tf.expand_dims(tf.expand_dims(self.b_v,0),0), [batch_size,self.num_n**2,1]))
        m_v_tplus1 = tf.reshape(m_v_tplus1,[batch_size,self.num_n,self.num_n,self.num_n])
        V_tplus1 = V_t + self.beta_v * m_v_tplus1
        return V_tplus1
    
    def message_passing_I(self, I_t, adj, demand, reuse_graph_tensor=False):
        """Message passing & update for currents
        Args:
            I_t: currents at timestep t, of size batch_size x N x N x N x N
            adj: adjacent matrix, of size batch_size x N x N
            demand: traffic demand as constant current source, of size batch_size x N x N x N x N
            reuse_graph_tensor: a boolean for if it is the 1st time calling this function
        Returns:
            I_t+1: currents at timestep t+1, of size batch_size x N x N x N
        """
        batch_size = tf.shape(adj)[0]
        if reuse_graph_tensor:
            self.w_i = tf.get_variable(name='w_i',
                    shape=[self.num_n**2,self.num_n**2],
                    dtype=tf.float32)
            self.b_i = tf.get_variable(name='b_i',
                    shape=[self.num_n**2],
                    dtype=tf.float32)
            I_t = tf.tile(tf.expand_dims(I_t,0), [batch_size,1,1,1,1])
        i_in = tf.matmul(tf.reshape(tf.transpose(I_t, [0,1,2,4,3]),[batch_size,self.num_n**3,self.num_n]), adj, transpose_b=True)
        i_in = tf.transpose(tf.reshape(i_in,[batch_size,self.num_n,self.num_n,self.num_n,self.num_n]), [0,1,2,4,3])
        i_in = tf.tile(tf.reduce_sum(i_in,-1,keep_dims=True),[1,1,1,1,self.num_n])
        i_out = tf.matmul(tf.reshape(I_t,[batch_size,self.num_n**3,self.num_n]), adj)
        i_out = tf.reshape(i_out,[batch_size,self.num_n,self.num_n,self.num_n,self.num_n])
        i_out = tf.tile(tf.reduce_sum(i_out,-2,keep_dims=True),[1,1,1,self.num_n,1])
        i_sum = i_in + i_out + demand # of size [batch_size, N, N, N, N]
        m_i_tplus1 = tf.tanh(tf.matmul(
            tf.reshape(i_sum, [batch_size,self.num_n**2,self.num_n**2]),
            tf.tile(tf.expand_dims(self.w_i,0), [batch_size,1,1]))
            + tf.tile(tf.expand_dims(tf.expand_dims(self.b_i,0),0), [batch_size,self.num_n**2,1]))
        m_i_tplus1 = tf.reshape(m_i_tplus1,[batch_size,self.num_n,self.num_n,self.num_n,self.num_n])
        I_tplus1 = self.beta_i * m_i_tplus1
        return I_tplus1

    def _egnn(self, obs):
         # adjacent matrix with edge features [b,N,N],node feature matrix [b,N,1]
        E_adj, X = tf.split(obs,[self.num_n,1],axis=-1)

        y = tf.reshape(X,[-1,1])
        # TODO:dropout?
        y = tf.matmul(y, self.gnn_weights[0])
        y += self.gnn_bias[0]
        y = tf.reshape(y,[-1,self.num_n,self.hid_dims[0]]) # [b,N,d0]
        y = tf.matmul(E_adj, y)
        #y = tf.sparse_tensor_dense_matmul(E_adj,y)
        y = tf.sigmoid(y)

        for l in range(1, len(self.gnn_weights)-1):
            y = tf.reshape(y,[-1,self.hid_dims[l-1]])
            y = tf.matmul(y, self.gnn_weights[l])
            y += self.gnn_bias[l]
            y = tf.reshape(y, [-1,self.num_n,self.hid_dims[l]])
            y = tf.matmul(E_adj, y) # [b,N,dl]
            y = tf.sigmoid(y)
        
        y = tf.reshape(y,[-1,self.hid_dims[-1]])
        y = tf.matmul(y, self.gnn_weights[-1])
        y += self.gnn_bias[-1]
        y = tf.reshape(y, [-1,self.num_n,self.out_dim])
        y = tf.matmul(E_adj, y)
        y = tf.sigmoid(y)

        return y

    def _para_init(self):
        """
        Initializing network parameters.
        """
        weights = []
        bias = []

        curr_in_dim = self.input_dim

        # hidden layers
        for hid_dim in self.hid_dims:
          weights.append(glorot([curr_in_dim, hid_dim]))
          bias.append(zeros([hid_dim]))
          curr_in_dim = hid_dim

        # output layer
        weights.append(glorot([curr_in_dim, self.out_dim]))
        bias.append(zeros([self.out_dim]))

        return weights, bias

def glorot(shape, dtype=tf.float32):
    # Xavier Glorot & Yoshua Bengio (AISTATS 2010) initialization (Eqn 16)
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    init = tf.random_uniform(
        shape, minval=-init_range, maxval=init_range, dtype=dtype)
    return tf.Variable(init)
        
def zeros(shape, dtype=tf.float32):
    init = tf.zeros(shape, dtype=dtype)
    return tf.Variable(init)

def mlp_extractor(flat_observations, net_arch, act_fun):
    """
    Constructs an MLP that receives observations as an input and outputs a latent representation for the policy and
    a value network. The ``net_arch`` parameter allows to specify the amount and size of the hidden layers and how many
    of them are shared between the policy network and the value network. It is assumed to be a list with the following
    structure:

    1. An arbitrary length (zero allowed) number of integers each specifying the number of units in a shared layer.
       If the number of ints is zero, there will be no shared layers.
    2. An optional dict, to specify the following non-shared layers for the value network and the policy network.
       It is formatted like ``dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])``.
       If it is missing any of the keys (pi or vf), no non-shared layers (empty list) is assumed.

    For example to construct a network with one shared layer of size 55 followed by two non-shared layers for the value
    network of size 255 and a single non-shared layer of size 128 for the policy network, the following layers_spec
    would be used: ``[55, dict(vf=[255, 255], pi=[128])]``. A simple shared network topology with two layers of size 128
    would be specified as [128, 128].

    :param flat_observations: (tf.Tensor) The observations to base policy and value function on.
    :param net_arch: ([int or dict]) The specification of the policy and value networks.
        See above for details on its formatting.
    :param act_fun: (tf function) The activation function to use for the networks.
    :return: (tf.Tensor, tf.Tensor) latent_policy, latent_value of the specified network.
        If all layers are shared, then ``latent_policy == latent_value``
    """
    latent = flat_observations
    policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
    value_only_layers = []  # Layer sizes of the network that only belongs to the value network

    # Iterate through the shared layers and build the shared parts of the network
    for idx, layer in enumerate(net_arch):
        if isinstance(layer, int):  # Check that this is a shared layer
            layer_size = layer
            latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
        else:
            assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
            if 'pi' in layer:
                assert isinstance(layer['pi'], list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                policy_only_layers = layer['pi']

            if 'vf' in layer:
                assert isinstance(layer['vf'], list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                value_only_layers = layer['vf']
            break  # From here on the network splits up in policy and value network

    # Build the non-shared part of the network
    latent_policy = latent
    latent_value = latent
    for idx, (pi_layer_size, vf_layer_size) in enumerate(zip_longest(policy_only_layers, value_only_layers)):
        if pi_layer_size is not None:
            assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
            latent_policy = act_fun(linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

        if vf_layer_size is not None:
            assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
            latent_value = act_fun(linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

    return latent_policy, latent_value
