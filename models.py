from layers import *
import tensorflow as tf
import tflearn

ENTROPY_WEIGHT = 1

class ActorNetwork(object):
    def __init__(self, sess, feature_dim, action_dim, learning_rate):
        self.sess = sess
        self.s_dim = feature_dim
        self.a_dim = action_dim
        self.lr_rate = learning_rate

        # Create the actor network
        self.adj, self.features, self.out = self.create_actor_network()

        # Get all network parameters
        self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Selected action, 0-1 vector
        self.acts = tf.placeholder(tf.float32, [None, self.a_dim])

        # This gradient will be provided by the critic network
        self.act_grad_weights = tf.placeholder(tf.float32, [None, 1])

        # Compute the objective (log action_vector and entropy)
        self.obj = tf.reduce_sum(tf.multiply(tf.log(tf.reduce_sum(tf.multiply(self.out, self.acts), 
                    reduction_indices=1, keep_dims=True)), -self.act_grad_weights)) \
                   + ENTROPY_WEIGHT * tf.reduce_sum(tf.multiply(self.out, tf.log(self.out + ENTROPY_EPS)))

        # Combine the gradients here
        self.actor_gradients = tf.gradients(self.obj, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))


    def create_actor_network(self):
        with tf.variable_scope('actor'):
            adj = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[0]])    # Node * Node
            features = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])   # Node * 7

            hidden1 = GCNLayer(output_dim=32)((adj, features))

            hidden2 = GCNLayer(output_dim=self.s_dim[0]*(self.s_dim[0]-1)/2 + 1)((adj, hidden1))

            edge_prob = tf.nn.softmax(hidden2[:self.s_dim[0]*(self.s_dim[0]-1)/2])
            edge = tf.argmax(edge_prob)
            stop = tf.nn.sigmoid(hidden2[-1])
            outputs = [node_1, node_2, stop]
            return adj, features, outputs

    def train(self, adj, features, acts, act_grad_weights):
        self.sess.run(self.optimize, feed_dict={
            self.adj: adj,
            self.features: features,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def predict(self, adj, features):
        return self.sess.run(self.out, feed_dict={
            self.adj: adj,
            self.features: features
        })

    def get_gradients(self, adj, features, acts, act_grad_weights):
        return self.sess.run(self.actor_gradients, feed_dict={
            self.adj: adj,
            self.features: features,
            self.acts: acts,
            self.act_grad_weights: act_grad_weights
        })

    def apply_gradients(self, actor_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.actor_gradients, actor_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })

class CriticNetwork(object):
    """
    Input to the network is the state and action, output is V(s).
    On policy: the action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, feature_dim, learning_rate):
        self.sess = sess
        self.s_dim = feature_dim
        self.lr_rate = learning_rate

        # Create the critic network
        self.inputs, self.out = self.create_critic_network()

        # Get all network parameters
        self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')

        # Set all network parameters
        self.input_network_params = []
        for param in self.network_params:
            self.input_network_params.append(tf.placeholder(tf.float32, shape=param.get_shape()))
        self.set_network_params_op = []
        for idx, param in enumerate(self.input_network_params):
            self.set_network_params_op.append(self.network_params[idx].assign(param))

        # Network target V(s)
        self.td_target = tf.placeholder(tf.float32, [None, 1])

        # Temporal Difference, will also be weights for actor_gradients
        self.td = tf.subtract(self.td_target, self.out)

        # Mean square error
        self.loss = tflearn.mean_square(self.td_target, self.out)

        # Compute critic gradient
        self.critic_gradients = tf.gradients(self.loss, self.network_params)

        # Optimization Op
        self.optimize = tf.train.RMSPropOptimizer(self.lr_rate).\
            apply_gradients(zip(self.critic_gradients, self.network_params))

    def create_critic_network(self):
        with tf.variable_scope('critic'):
            adj = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[0]])    # Node * Node
            features = tflearn.input_data(shape=[None, self.s_dim[0], self.s_dim[1]])   # Node * 7

            hidden1 = GraphConvolution(input_dim=self.s_dim[1],
                                       output_dim=16,
                                       adj=self.adj,
                                       act=tf.nn.relu,
                                       dropout=self.dropout)(features)

            hidden2 = GraphConvolution(input_dim=16,
                                       output_dim=3,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(hidden1)

            outputs = tf.nn.softmax(hidden2)
            return adj, features, outputs

    def train(self, adj, features, td_target):
        return self.sess.run([self.loss, self.optimize], feed_dict={
            self.adj: adj,
            self.features: features,
            self.td_target: td_target
        })

    def predict(self, adj, features):
        return self.sess.run(self.out, feed_dict={
            self.adj: adj,
            self.features: features
        })

    def get_td(self, adj, features, td_target):
        return self.sess.run(self.td, feed_dict={
            self.adj: adj,
            self.features: features,
            self.td_target: td_target
        })

    def get_gradients(self, adj, features, td_target):
        return self.sess.run(self.critic_gradients, feed_dict={
            self.adj: adj,
            self.features: features,
            self.td_target: td_target
        })

    def apply_gradients(self, critic_gradients):
        return self.sess.run(self.optimize, feed_dict={
            i: d for i, d in zip(self.critic_gradients, critic_gradients)
        })

    def get_network_params(self):
        return self.sess.run(self.network_params)

    def set_network_params(self, input_network_params):
        self.sess.run(self.set_network_params_op, feed_dict={
            i: d for i, d in zip(self.input_network_params, input_network_params)
        })


def compute_gradients(adj_batch, f_batch, a_batch, r_batch, terminal, actor, critic):
    """
    batch of s, a, r is from samples in a sequence
    the format is in np.array([batch_size, s/a/r_dim])
    terminal is True when sequence ends as a terminal state
    """
    assert adj_batch.shape[0] == f_batch.shape[0]
    assert adj_batch.shape[0] == a_batch.shape[0]
    assert adj_batch.shape[0] == r_batch.shape[0]
    ba_size = adj_batch.shape[0]

    v_batch = critic.predict(adj_batch, f_batch)

    R_batch = np.zeros(r_batch.shape)

    if terminal:
        R_batch[-1, 0] = 0  # terminal state
    else:
        R_batch[-1, 0] = v_batch[-1, 0]  # boot strap from last state

    for t in reversed(range(ba_size - 1)):
        R_batch[t, 0] = r_batch[t] + GAMMA * R_batch[t + 1, 0]

    td_batch = R_batch - v_batch

    actor_gradients = actor.get_gradients(adj_batch, f_batch, a_batch, td_batch)
    critic_gradients = critic.get_gradients(adj_batch, f_batch, R_batch)

    return actor_gradients, critic_gradients, td_batch


def build_summaries():
    td_loss = tf.Variable(0.)
    tf.summary.scalar("TD_loss", td_loss)
    eps_total_reward = tf.Variable(0.)
    tf.summary.scalar("Eps_total_reward", eps_total_reward)
    avg_entropy = tf.Variable(0.)
    tf.summary.scalar("Avg_entropy", avg_entropy)

    summary_vars = [td_loss, eps_total_reward, avg_entropy]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
