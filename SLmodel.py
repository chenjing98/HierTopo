import numpy as np
import random
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
        
        print("Model built.")
    
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
        self.adj_ph = tf.placeholder(tf.float32,shape=[None, self.n_nodes, self.n_nodes],name="adj_ph")
        self.deg_ph = tf.placeholder(tf.float32,shape=[None, self.n_nodes],name="deg_ph")
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

    def data_next_batch(self, idx):
        demands = []
        adjs = []
        degrees = []
        vs = []
        for ind in idx:
            demand = self.dataset['demand'][ind]
            adj = self.dataset['adj'][ind]
            degree = self.dataset['degree'][ind]
            v = self.dataset['v'][ind]
            demands.append(demand[np.newaxis,:])
            adjs.append(adj[np.newaxis,:])
            degrees.append(degree[np.newaxis,:])
            vs.append(v[np.newaxis,:])
        batch_demand = np.concatenate(tuple(demands), axis=0)
        batch_adj = np.concatenate(tuple(adjs), axis=0)
        batch_deg = np.concatenate(tuple(degrees), axis=0)
        batch_v = np.concatenate(tuple(vs), axis=0)
        return batch_demand, batch_adj, batch_deg, batch_v

    def train(self, dataset_file, num_data, train_steps=10000000, save_step=100000, tb_log_interval=100):
        # Load dataset
        self.dataset = np.load(dataset_file, allow_pickle=True)
        ind_sampler = iter(BatchSampler(num_data, self.batch_size, False))
        
        # Start training
        print("Training started.")
        for step in range(train_steps):
            next_idx = next(ind_sampler)
            batch_demand, batch_adj, batch_deg, batch_y = self.data_next_batch(next_idx)
            _, loss = self.sess.run([self.optim_op, self.loss], feed_dict={self.demand_ph: batch_demand,
                                                                           self.adj_ph: batch_adj,
                                                                           self.deg_ph: batch_deg,
                                                                           self.label_ph: batch_y})
            if self.enable_summary_writer and (step + 1) % tb_log_interval == 0:
                summary_str = self.sess.run(self.merged_summary_op)
                self.summary_writer.add_summary(summary_str, step)
            if (step + 1) % 10 == 0:
                print("Step: {0} | Loss: {1:.7f}".format(step + 1, loss))
            if step > 0 and (step + 1) % save_step == 0:
                if self.enable_saver:
                    self.save(step)
                print("Model saved.")
        print("Training terminated.")

    def predict(self, demand, adj, deg):
        return self.sess.run([self.nn_output], feed_dict={self.demand_ph: demand,
                                                          self.adj_ph: adj,
                                                          self.deg_ph: deg})
    

    def save(self, step):
        self.saver.save(self.sess, self.save_path)
        print("Step: {} | Saver reach checkpoint.".format(step + 1))

    def load_model(self, model):
        pass

class BatchSampler(object):
    """BatchSampler to yield a mini-batch of indices.
    Original code from https://pytorch.org/docs/stable/_modules/torch/utils/data/sampler.html
    
    Args:
        datasource (npz file): Dataset.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, num_data, batch_size, drop_last):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        
        self.num_data = num_data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = list(range(self.num_data))

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch
        self.shuffle()

    def __len__(self):
        if self.drop_last:
            return self.num_data // self.batch_size
        else:
            return (self.num_data + self.batch_size - 1) // self.batch_size

    def shuffle(self):
        random.shuffle(self.sampler)

def main():
    with tf.Session() as sess:
        train_model = supervisedModel(sess,8,4,[3,64,1],batch_size=32,save_path="./model",summary_path="./summary")
        train_model.train("./dataset_8_64000.npz", 64000)

if __name__ == "__main__":
    main()