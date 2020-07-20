import tensorflow as tf

from .aggregator import TwoMaxLayerPoolingAggregator

class model(object):
    def __init__(self, num_n, max_degree, dims, dropout=.1, concat=False, **kwargs):
        self.num_n = num_n
        self.max_degree = max_degree + 1
        self.aggregator_cls = TwoMaxLayerPoolingAggregator
        
        self.dims = dims
        self.concat = concat
        self.dropout = dropout

        self._build()

    def _build(self):
        # build aggregators
        self.n2e_aggregators, self.e2n_aggregators = self._aggregate()
    
    def forward(self, adj, demand, available_degrees):
        init_node_features = self.single_node_features(adj, available_degrees)
        init_edge_features = self.single_edge_features(demand)
        
        node_features = init_node_features # [batch_size, N, dim]
        edge_features = init_edge_features # [batch_size, N, N+1, dim]

        for i in range(len(self.dims)-1):
            node_neigh_features = self.pad_neighbor_node_features(node_features) #[batch_size, N, N+1, 2, dim]
            edge_neigh_features = self.pad_neighbor_edge_features(adj, edge_features) #[batch_size, N, max_degree, dim]
            curr_edge_features = self.edge_iterations(self.n2e_aggregators[i],edge_features,node_neigh_features)
            curr_node_features = self.node_iterations(self.e2n_aggregators[i],node_features,edge_neigh_features)
            node_features = curr_node_features
            edge_features = curr_edge_features
        
        #output_features = self.demands_sumup(node_features)
        output_features = tf.squeeze(node_features,axis=[-1])
        return output_features
    
    def single_node_features(self, adj, available_degrees):
        """Abstracting node features.
        
        Args:
            adj: Adjacency matrix of size [batch_size, N, N]
            available_degrees: vector for available degrees of size [batch_size, N]

        Returns:
            node_features: [batch_size, N, dim], 
                dim=2,including available degrees, current_degrees
        """
        
        curr_degree = tf.reduce_sum(adj,axis=-1,keepdims=True)
        expand_avail_degree = tf.expand_dims(available_degrees,-1) # [batch_size, N, 1]
        features = [expand_avail_degree, curr_degree]
        node_features = tf.concat(features,-1)
        return node_features

    def single_edge_features(self, demand):
        """Abstracting edge features.

        Args:
            demand: traffic demand matrix of size [batch_size, N, N]
        
        Returns:
            edge_features: [batch_size, N, N+1, dim],
                dim=2,including bi-directional flow bandwidth on each link
                (N+1 is for n network nodes and 1 ground node)
        """
        zero_vec = tf.expand_dims(tf.zeros_like(demand),-1)
        edge_features_net = tf.tile(zero_vec, [1, 1, 1, 2])# [batch_size, N, N, 2]
        demand_sums = [tf.expand_dims(tf.reduce_sum(demand, -2),-1), 
                       tf.reduce_sum(demand, -1, keepdims=True)]
        edge_features_2ground = tf.expand_dims(tf.concat(demand_sums, -1), -2) # [bathc_size, N, 1, 2]
        features = [edge_features_net, edge_features_2ground]
        edge_features = tf.concat(features, -2)
        return edge_features

    def pad_neighbor_node_features(self, node_features):
        """Padding neighbor features from edge features to node features
        
        Args:
            node_features: of size [batch_size, N, dim]

        Returns:
            neighbor_node_features: of size [batch_size, N, N+1, 2, dim]
        """
        features_inrow = tf.tile(tf.expand_dims(node_features,1),[1,self.num_n,1,1])
        features_incol = tf.tile(tf.expand_dims(node_features,2),[1,1,self.num_n,1])
        netnode_features_list = [tf.expand_dims(features_inrow,-2), tf.expand_dims(features_incol,-2)]
        netnode_features = tf.concat(netnode_features_list,-2) # [batch_size, N, N, 2, dim]

        features_2ground = tf.ones_like(node_features, tf.float32) * self.num_n
        ground_features_list = [tf.expand_dims(node_features,-2), tf.expand_dims(features_2ground,-2)] 
        ground_features = tf.concat(ground_features_list, -2) # [batch_size, N, 2, dim]

        features_list = [netnode_features, tf.expand_dims(ground_features,-3)]
        neighbor_node_features = tf.concat(features_list, -3)
        return neighbor_node_features

    def pad_neighbor_edge_features(self, adj, edge_features):
        """Padding neighbor features into MLP input dimension (max_degree) from node features to edge features
        Assume that all degrees <= max_degree.
        
        Args:
            adj: Adjacency matrix of size [batch_size, N, N]
            edge_features: of size [batch_size, N, N+1, dim]

        Returns:
            neighbor_features: of size [batch_size, N, max_degree, dim]
        """
        batch_size = tf.shape(edge_features)[0]
        dim = tf.shape(edge_features)[-1]
        edge_features_net, edge_features_ground = tf.split(edge_features, [self.num_n, 1], axis=-2)

        zeros = tf.zeros_like(tf.reduce_max(edge_features,-2,keepdims=True), tf.float32)
        zeros = tf.tile(zeros, [1,1,self.max_degree-1,1]) # [batch_size,N,max_degree-1,dim]
        expanded_nfeatures = tf.concat([edge_features_net,zeros],axis=-2) # [batch_size,N,N+max_degree-1,dim]
        expanded_adj = self.expand_deg_adj(adj) # [batch_size,N,N+max_degree-1]
        neighbor_inds = tf.where(expanded_adj)
        neighbor_features_net = tf.gather_nd(expanded_nfeatures,neighbor_inds) # [batch_size*N*(max_degree-1),dim]
        neighbor_features_net = tf.reshape(neighbor_features_net,[batch_size,self.num_n,self.max_degree-1,dim]) # [batch_size,N,max_degree-1,dim]
        
        neighbor_features_list = [neighbor_features_net, edge_features_ground]
        neighbor_features = tf.concat(neighbor_features_list, -2)
        return neighbor_features

    def _aggregate(self):
        # length: number of layers + 1
        dims = self.dims
        n2e_aggregators = []
        e2n_aggregators = []
        for layer in range(len(dims)-1):
            dim_mult = 2 if self.concat and (layer != 0) else 1
            # aggregator at current layer
            if layer == len(dims) - 2:
                n2e_aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], 2,
                        act=lambda x : x, dropout=self.dropout, concat=self.concat)
                e2n_aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], self.max_degree,
                        act=lambda x : x, dropout=self.dropout, concat=self.concat)
            elif layer==0:
                n2e_aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], 2,
                        dropout=0.0, concat=self.concat)
                e2n_aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], self.max_degree,
                        dropout=0.0, concat=self.concat)
            else:
                n2e_aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], 2,
                        dropout=self.dropout, concat=self.concat)
                e2n_aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], self.max_degree,
                        dropout=self.dropout, concat=self.concat)
            n2e_aggregators.append(n2e_aggregator)
            e2n_aggregators.append(e2n_aggregator)    
        return n2e_aggregators, e2n_aggregators
        

    def edge_iterations(self, aggregator, e_features_tminus1, n_neighbor_features_tminus1):
        """
        Args:
            e_features_tminus1: of size [batch_size, N, N+1, dim_in]
            n_neighbor_features_tminus1: of size [batch_size, N, N+1, 2, dim_in]
        """
        batch_size = tf.shape(e_features_tminus1)[0]
        dim_in = aggregator.input_dim
        self_features = tf.reshape(e_features_tminus1,\
            [batch_size*self.num_n*(self.num_n+1),dim_in])
        neigh_features = tf.reshape(n_neighbor_features_tminus1,\
            [batch_size*self.num_n*(self.num_n+1),2,dim_in])
        f_t = aggregator.call((self_features,neigh_features)) # [batch_size x N x (N+1), dim_out]
        e_features_t = tf.reshape(f_t, [batch_size,self.num_n,self.num_n+1,aggregator.output_dim])
        return e_features_t

    def node_iterations(self, aggregator, n_features_tminus1, e_neighbor_features_tminus1):
        """
        Args:
            n_features_tminus1: of size [batch_size, N, dim_in]
            e_neighbor_features_tminus1: of size [batch_size, N, max_degree, dim_in]
        """
        batch_size = tf.shape(n_features_tminus1)[0]
        dim_in = aggregator.input_dim
        self_features = tf.reshape(n_features_tminus1,\
            [batch_size*self.num_n,dim_in])
        neigh_features = tf.reshape(e_neighbor_features_tminus1,\
            [batch_size*self.num_n,self.max_degree,dim_in])
        f_t = aggregator.call((self_features,neigh_features)) # [batch_size x N, dim_out]
        n_features_t = tf.reshape(f_t, [batch_size,self.num_n,aggregator.output_dim])
        return n_features_t

    def demands_sumup(self, node_features):
        """
        Args:
            node_features: of size [batch_size, N, N, N, output_dim]
        
        Returns:
            node features (after suming up for all demands) : of size [batch_size, N, outdim]
        """
        batch_size = tf.shape(node_features)[0]
        dim_mult = 2 if self.concat else 1
        features = tf.reshape(node_features,[batch_size,self.num_n**2, self.num_n, dim_mult*self.dims[-1]])
        features = tf.reduce_sum(features,axis=1)
        return features
    
    def expand_deg_adj(self, adj):
        """
        Args:
            adj: adjacency matrix [batch_size,N,N]
        
        Returns:
            expanded_adj: [batch_size,N, N + max_degree], 
                each line has max_degree of 1s, 
                [:,:N] is exactly the input adj
        """
        deg_inuse = tf.reduce_sum(adj,axis=-1) # [batch_size,N]
        deg_pad = self.max_degree - 1 - deg_inuse # [batch_size,N]
        pad_cols = []
        ones = tf.ones_like(tf.tile(tf.expand_dims(deg_inuse,-1),[1,1,self.max_degree-1]),tf.float32)
        zeros = tf.zeros_like(ones,tf.float32)
        for i in range(self.max_degree - 1):
            col_i = tf.expand_dims((deg_pad > i),-1)
            pad_cols.append(col_i)
        pad_pos = tf.concat(pad_cols,-1) # [batch_size,N,max_degree-1]
        paddings = tf.where(pad_pos,ones,zeros)
        padded_adj = tf.concat([adj,paddings],axis=-1) #[batch_size,N,N+max_degree-1]
        return padded_adj

    def solid_feature(self, edge_features):
        """
        Args:
            edge_features: of size [batch_size, N, N+1, dim]
        Returns:
            reset features on links to the ground...
        """
        pass
