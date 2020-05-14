import numpy as np
import tensorflow as tf
from GNN.GNN_nodeedge import model

class supervisedModel(object):
    """
    A supervised learning model for network topology adjusting.

    Reference codes are from https://github.com/Ceruleanacg/Personae

    :param sess: (tf.Session) tensorflow session
    :param n_nodes: (int) the number of nodes in the network
    :param max_degree: (int) the maximum node degree (for each switch)
    :param dims: ([int]) dimensions for GNN aggregators (must begin with 3 and end with 1)
    :param batch_size: (int) batchsize for training
    :param learning_rate: (float) learning rate
    :param adam_epsilon: (float)
    :param reuse: (bool) whether to reuse the aggregators among nodes
    :param enable_saver: (bool) whether to save model periodly
    :param save_path: (str) the directory for saving the model
    :param enable_summary_writer: (bool) whether to log data onto tensorboard
    :param summary_path: (str) the directory for saving tensorboard logs
    """

    def __init__(self, sess, n_nodes, max_degree, dims, 
                 batch_size=64, learning_rate=1e-4, adam_epsilon=1e-8, reuse=True,
                 enable_saver=True, save_path=None,
                 enable_summary_writer=True, summary_path=None):

        self.sess = sess

        self.n_nodes = n_nodes
        self.max_degree = max_degree
        self.dims = dims
        self.reuse = reuse
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon

        self.enable_saver = enable_saver
        self.enable_summary_writer = enable_summary_writer
        self.save_path = save_path
        self.summary_path = summary_path
        
        self._init_placeholder()
        self._init_nn()
        self._init_op()
        self._init_saver()
        self._init_summary_writer()
        
    
    def _init_placeholder(self):
        """
        Initialize the placeholders:
        - demand_ph: traffic demand placeholder
        - adj_ph: adjacency matrix placeholder
        - deg_ph: available node degreee placeholder
        - label_ph: potentials for the optimal action
        """
        # inputs
        self.demand_ph = tf.placeholder(tf.float32,shape=[None, self.n_nodes, self.n_nodes],name="demand_ph")
        self.adj_ph = tf.placeholder(tf.int8,shape=[None, self.n_nodes, self.n_nodes],name="adj_ph")
        self.deg_ph = tf.placeholder(tf.int8,shape=[None, self.n_nodes],name="deg_ph")
        # label
        self.label_ph = tf.placeholder(tf.float32,shape=[None, self.n_nodes],name="label_ph")

    def _init_nn(self):
        """
        Initialize the neural network.
        """
        with tf.variable_scope("gnn", reuse=self.reuse):
            GNNnet = model(self.n_nodes,self.max_degree,self.dims)
            self.nn_output = GNNnet.forward(self.adj_ph,self.demand_ph,self.deg_ph)
            if self.enable_summary_writer:
                tf.summary.histogram('potential', self.nn_output)

    def _init_op(self):
        """
        Define the loss and the optimizer.
        """
        with tf.variable_scope('loss'):
            self.loss = tf.losses.mean_squared_error(self.label_ph, self.nn_output)
            if self.enable_summary_writer:
                tf.summary.scalar('mean', tf.reduce_mean(self.loss))
        
        with tf.variable_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=self.adam_epsilon)
            self.optim_op = self.optimizer.minimize(self.loss)
        
        self.sess.run(tf.global_variables_initializer())

    def _init_saver(self):
        if self.enable_saver:
            self.saver = tf.train.Saver()

    def _init_summary_writer(self):
        if self.enable_summary_writer:
            self.merged_summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(self.summary_path, self.sess.graph)

    def _init_dataloader(self, dataset_file, batch_size, shuffle_buf=10000):
        # Load dataset
        data = np.load(dataset_file, allow_pickle=True)
        dataset = tf.data.Dataset.from_tensor_slices(
            (data['demand'], data['adj'], data['degree'], data['v'])
        )
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(shuffle_buf)
        data_iter = dataset.make_one_shot_iterator()
        return data_iter

    def train(self, dataset_file, train_steps=10000000, save_step=100000, tb_log_interval=100):
        data_loader = self._init_dataloader(dataset_file,self.batch_size)
        # Start training
        for step in range(train_steps):
            batch_demand, batch_adj, batch_deg, batch_y = data_loader.get_next()
            _, loss = self.sess.run([self.optim_op, self.loss], feed_dict={self.demand_ph: batch_demand,
                                                                           self.adj_ph: batch_adj,
                                                                           self.deg_ph: batch_deg,
                                                                           self.label_ph: batch_y})
            if self.enable_summary_writer and (step + 1) % tb_log_interval == 0:
                summary_str = self.sess.run(self.merged_summary_op)
                self.summary_writer.add_summary(summary_str, step)
            if (step + 1) % 1000 == 0:
                print("Step: {0} | Loss: {1:.7f}".format(step + 1, loss))
            if step > 0 and (step + 1) % save_step == 0:
                if self.enable_saver:
                    self.save(step)
        
    def predict(self, demand, adj, deg):
        return self.sess.run([self.nn_output], feed_dict={self.demand_ph: demand,
                                                          self.adj_ph: adj,
                                                          self.deg_ph: deg})
    

    def save(self, step):
        self.saver.save(self.sess, self.save_path)
        print("Step: {} | Saver reach checkpoint.".format(step + 1))

    def load_model(self, model):
        pass
