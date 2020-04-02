import tensorflow as tf
import numpy as np


_LAYER_UIDS = {}

def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def matmul(x, y, sparse=False):
    """Wrapper for sparse matrix multiplication."""
    if sparse:
        return tf.sparse_tensor_dense_matmul(x, y)
    return tf.matmul(x, y)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    # Properties
        name: String, defines the variable scope of the layer.
    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GCNLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, activation=tf.nn.relu, **kwargs):
        self.output_dim = output_dim
        self.activation = activation
        super().__init__(**kwargs)

    def build(self, input_shape):   # [batch_size, num_vertices, num_vertices], [batch_size, num_vertices, num_features]
        A_shape, H_shape = input_shape
        self.num_vertices = A_shape[1].value
        self.W = self.add_weight(   # [num_features, output_dim]
            name='W',
            shape=[H_shape[2].value, self.output_dim]
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs:  A for adjacent matrix [batch_size, num_vertices, num_vertices] (should be normalized in advance)
                        H for features [batch_size, num_vertices, num_features]
        """
        A, H = inputs[0], inputs[1]
        # A * H * W [batch_size, num_vertices, num_vertices] * [batch_size, num_vertices, num_features] * [num_features, output_dim]
        # see https://www.tensorflow.org/api_docs/python/tf/tensordot and https://www.machenxiao.com/blog/tensordot
        # for tf.tensordot()
        H_next = tf.tensordot(tf.matmul(A, H), self.W, axes=[2, 0])
        if self.activation is not None:
            H_next = self.activation(H_next)
        return H_next
