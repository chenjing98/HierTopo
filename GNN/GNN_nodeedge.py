import tensorflow as tf

from GNN.aggregator import TwoMaxLayerPoolingAggregator

class model(object):
    def __init__(self, num_n, max_degree, batch_size, dropout=.4, concat=False, **kwargs):
        self.num_n = num_n
        self.max_degree = max_degree
        self.aggregator_cls = TwoMaxLayerPoolingAggregator
        
        self.dim = 4
        self.depths = 10
        self.concat = concat
        self.batch_size = batch_size
        
        self.dropout = dropout

        self._build()

    def _build(self):
        # build aggregators
        self.n2e_aggregator, self.e2n_aggregator = self._aggregate()
    
    def forward(self, adj, demand, available_degrees):
        init_node_features = self.single_node_features(adj, demand, available_degrees)
        init_edge_features = self.single_edge_features(adj)
        
        node_features = init_node_features
        edge_features = init_edge_features

        for _ in range(self.depths):
            node_neigh_features = self.pad_neighbor_features_e2n(node_features) #[batch_size, N, N, N, N, 2, dim]
            edge_neigh_features = self.pad_neighbor_features_n2e(adj, edge_features) #[batch_size, N, N, N, max_degree, dim]
            curr_edge_features = self.n2e_iterations(self.n2e_aggregator,edge_features,node_neigh_features)
            curr_node_features = self.e2n_iterations(self.e2n_aggregator,node_features,edge_neigh_features)
            node_features = curr_node_features
            edge_features = curr_edge_features
        
        output_features = self.demands_sumup(node_features,self.dim)
        return output_features
    
    def single_node_features(self, adj, demand, available_degrees):
        """Abstracting node features.
        
        Args:
            adj: Adjacency matrix of size [batch_size, N, N]
            demand: demand matrix of size [batch_size, N, N]
            available_degrees: vector for available degrees of size [batch_size, 1, N]

        Returns:
            node_features: [batch_size, N, N, N, dim], 
                dim=4,including available degrees, current degrees, out_demand, in_demand
        """

        batch_size = self.batch_size
        #batch_size = tf.shape(demand)[0]
        expand_demand = tf.tile(tf.expand_dims(demand, -1),[1,1,1,self.num_n])
        I = tf.eye(self.num_n,batch_shape=tf.expand_dims(batch_size,0))
        absrow = tf.tile(tf.expand_dims(I,2),[1,1,self.num_n,1])
        abscol = tf.tile(tf.expand_dims(I,1),[1,self.num_n,1,1])
        out_demand = tf.multiply(expand_demand, absrow)
        in_demand = tf.multiply(expand_demand, abscol)
        expand_avail_degree = tf.tile(tf.expand_dims(available_degrees,1),[1,self.num_n,self.num_n,1])
        current_degrees = tf.reduce_sum(adj,axis=-1)
        expand_curr_degree = tf.tile(tf.expand_dims(tf.expand_dims(current_degrees,1),1),[1,self.num_n,self.num_n,1])
        features = [tf.expand_dims(expand_avail_degree,-1),tf.expand_dims(expand_curr_degree,-1),\
            tf.expand_dims(out_demand,-1),tf.expand_dims(in_demand,-1)]
        node_features = tf.concat(features,-1)
        return node_features

    def single_edge_features(self, adj):
        """Abstracting edge features.

        Args:
            adj: Adjacency matrix of size [batch_size, N, N]
        
        Returns:
            edge_features: [batch_size, N, N, N, N, dim],
                dim=4,including available degrees, current degrees, out_demand, in_demand
                (actually same for each flow)
        """
        current_degrees = tf.expand_dims(2 * adj, -1) # [batch_size, N, N, 1]
        available_degrees = 2 * tf.ones_like(current_degrees)
        out_demand = tf.zeros_like(current_degrees)
        in_demand = tf.zeros_like(current_degrees)
        features = [available_degrees, current_degrees, out_demand, in_demand]
        edge_features_perflow = tf.concat(features, -1)
        edge_features = tf.tile(tf.expand_dims(tf.expand_dims(edge_features_perflow,1), 1),[1,self.num_n, self.num_n, 1,1,1])
        return edge_features

    def pad_neighbor_features_e2n(self, node_features):
        """Padding neighbor features from edge features to node features
        
        Args:
            node_features: of size [batch_size, N, N, N, dim]

        Returns:
            neighbor_features: of size [batch_size, N, N, N, N, 2, dim]
        """
        features_inrow = tf.tile(tf.expand_dims(node_features,3),[1,1,1,self.num_n,1,1])
        features_incol = tf.tile(tf.expand_dims(node_features,4),[1,1,1,1,self.num_n,1])
        features = [tf.expand_dims(features_inrow,-2), tf.expand_dims(features_incol,-2)]
        neighbor_node_features = tf.concat(features,-2)
        return neighbor_node_features

    def pad_neighbor_features_n2e(self, adj, edge_features):
        """Padding neighbor features into MLP input dimension (max_degree) from node features to edge features
        Assume that all degrees <= max_degree.
        
        Args:
            adj: Adjacency matrix of size [batch_size, N, N]
            edge_features: of size [batch_size, N, N, N, N, dim]

        Returns:
            neighbor_features: of size [batch_size, N, N, N, max_degree, dim]
        """
        dim = tf.shape(edge_features)[-1]
        batch_neighbor_features_list = []
        for batch_no in range(self.batch_size):
            batch_adj = tf.gather(adj,batch_no,axis=0)
            batch_edge_features = tf.gather(edge_features,batch_no,axis=0) # [N,N,N,N,dim]
            zeros = tf.zeros_like(
                tf.tile(tf.reduce_sum(batch_edge_features,-2,keepdims=True),[1,1,1,self.max_degree,1]),
                tf.float32) # [N,N,N,max_degree,dim]
            expanded_nfeatures = tf.transpose(tf.concat([batch_edge_features,zeros],axis=-2),[2,3,0,1,4]) # [N,N+max_degree,N,N,dim]
            expanded_adj = self.expand_deg_adj(batch_adj) # [N, N+max_degree]
            neighbor_inds = tf.where(expanded_adj)
            neighbor_features = tf.gather_nd(expanded_nfeatures,neighbor_inds) # [N*max_degree,N,N,dim]
            neighbor_features = tf.transpose(
                tf.reshape(neighbor_features,[self.num_n,self.max_degree,self.num_n,self.num_n,dim]),
                [2,3,0,1,4]) # [N,N,N,max_degree,dim]
            batch_neighbor_features_list.append(tf.expand_dims(neighbor_features,0))
        batch_neighbor_features = tf.concat(batch_neighbor_features_list,0)
        return batch_neighbor_features

    def _aggregate(self):
        """
        # length: number of layers + 1
        aggregators = []
        for layer in range(len(dims)-1):
            dim_mult = 2 if self.concat and (layer != 0) else 1
            # aggregator at current layer
            if layer == len(dims) - 2:
                aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], self.max_degree,
                        act=lambda x : x, dropout=self.dropout, concat=self.concat)
            elif layer==0:
                aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], self.max_degree,
                        dropout=0.0, concat=self.concat)
            else:
                aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], self.max_degree,
                        dropout=self.dropout, concat=self.concat)
            aggregators.append(aggregator)
            
        return aggregators
        """
        n2e_aggregator = self.aggregator_cls(self.dim, self.dim, 2,
                        dropout=0.0, concat=self.concat)
        e2n_aggregator = self.aggregator_cls(self.dim, self.dim, self.max_degree,
                        dropout=0.0, concat=self.concat)
        return n2e_aggregator, e2n_aggregator

    def n2e_iterations(self, aggregator, e_features_tminus1, n_neighbor_features_tminus1):
        """
        Args:
            e_features_tminus1: of size [batch_size, N, N, N, N, dim_in]
            n_neighbor_features_tminus1: of size [batch_size, N, N, N, N, 2, dim_in]
        """
        #batch_size = tf.shape(features_tminus1)[0]
        batch_size = self.batch_size
        dim_in = aggregator.input_dim
        self_features = tf.reshape(e_features_tminus1,\
            [batch_size*self.num_n*self.num_n,self.num_n**2,dim_in])
        neigh_features = tf.reshape(n_neighbor_features_tminus1,\
            [batch_size*self.num_n*self.num_n,self.num_n**2,2,dim_in])
        features_t_list = []
        # Iterates among edges
        for v in range(self.num_n**2):
            self_fv = tf.gather(self_features,v,axis=1)
            neigh_fv = tf.gather(neigh_features,v,axis=1)
            fv_t = aggregator.call((self_fv,neigh_fv)) # [batch_size x N^2, dim_out]
            fv_t = tf.expand_dims(fv_t,1)
            features_t_list.append(fv_t)
        features_t = tf.concat(features_t_list,1)
        e_features_t = tf.reshape(features_t, [batch_size,self.num_n,self.num_n,self.num_n,self.num_n,aggregator.output_dim])
        return e_features_t

    def e2n_iterations(self, aggregator, n_features_tminus1, e_neighbor_features_tminus1):
        """
        Args:
            n_features_tminus1: of size [batch_size, N, N, N, dim_in]
            e_neighbor_features_tminus1: of size [batch_size, N, N, N, max_degree, dim_in]
        """
        #batch_size = tf.shape(features_tminus1)[0]
        batch_size = self.batch_size
        dim_in = aggregator.input_dim
        self_features = tf.reshape(n_features_tminus1,\
            [batch_size*self.num_n*self.num_n,self.num_n,dim_in])
        neigh_features = tf.reshape(e_neighbor_features_tminus1,\
            [batch_size*self.num_n*self.num_n,self.num_n,self.max_degree,dim_in])
        features_t_list = []
        # Iterates among nodes
        for v in range(self.num_n):
            self_fv = tf.gather(self_features,v,axis=1)
            neigh_fv = tf.gather(neigh_features,v,axis=1)
            fv_t = aggregator.call((self_fv,neigh_fv)) # [batch_size x N^2, dim_out]
            fv_t = tf.expand_dims(fv_t,1)
            features_t_list.append(fv_t)
        features_t = tf.concat(features_t_list,1)
        n_features_t = tf.reshape(features_t, [batch_size,self.num_n,self.num_n,self.num_n,aggregator.output_dim])
        return n_features_t

    def demands_sumup(self, node_features, outdim):
        """
        Args:
            node_features: of size [batch_size, N, N, N, output_dim]
        
        Returns:
            node features (after suming up for all demands) : of size [batch_size, N, outdim]
        """
        batch_size = tf.shape(node_features)[0]
        features = tf.reshape(node_features,[batch_size,self.num_n**2, self.num_n, outdim])
        features = tf.reduce_sum(features,axis=1)
        return features
    
    def expand_deg_adj(self,adj):
        """
        Args:
            adj: adjacency matrix [N,N]
        
        Returns:
            expanded_adj: [N, N + max_degree], 
                each line has max_degree of 1s, 
                [:,:N] is exactly the input adj
        """
        deg_inuse = tf.reduce_sum(adj,axis=-1) # [N,]
        deg_pad = self.max_degree - deg_inuse # [N, ]
        pad_cols = []
        ones = tf.ones((self.num_n,self.max_degree),tf.float32)
        zeros = tf.zeros((self.num_n,self.max_degree),tf.float32)
        for i in range(self.max_degree):
            col_i = tf.expand_dims((deg_pad > i),-1)
            pad_cols.append(col_i)
        pad_pos = tf.concat(pad_cols,-1)
        paddings = tf.where(pad_pos,ones,zeros)
        padded_adj = tf.concat([adj,paddings],axis=-1)
        return padded_adj
