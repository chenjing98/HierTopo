import tensorflow as tf

from GNN.aggregator import TwoMaxLayerPoolingAggregator

class model(object):
    def __init__(self, num_n, max_degree, batch_size, dims, dropout=.4, concat=True, **kwargs):
        self.num_n = num_n
        self.max_degree = max_degree
        self.aggregator_cls = TwoMaxLayerPoolingAggregator
        
        self.dims = dims # input dim must be 4
        self.concat = concat
        self.batch_size = batch_size
        
        self.dropout = dropout

        self._build()

    def _build(self):
        # build aggregators
        self.aggregators = self._aggregate(self.dims)
    
    def forward(self, adj, demand, available_degrees):
        init_node_features = self.single_node_features(adj, demand, available_degrees)
        init_neigh_features = self.pad_neighbor_features(adj, init_node_features)
        curr_node_features = init_node_features
        curr_neigh_features = init_neigh_features

        for aggregator in self.aggregators:
            curr_node_features = self.node_iterations(aggregator,curr_node_features,curr_neigh_features)
            curr_neigh_features = self.pad_neighbor_features(adj,curr_node_features)
        
        output_features = self.demands_sumup(curr_node_features)
        return output_features
        
    def single_node_features(self, adj, demand, available_degrees):
        """Abstracting node features.
        
        Args:
            adj: Adjacency matrix of size [batch_size x N x N]
            demand: demand matrix of size [batch_size x N x N]
            available_degrees: vector for available degrees of size [batch_size x N]
        Returns:
            node_features: [batch_size x N x N x N x dim], 
                            dim=4,including available degrees, current degrees, out_demand, in_demand
        """
        batch_size = tf.shape(demand)[0]
        expand_demand = tf.tile(tf.expand_dims(demand, -1),[1,1,1,self.num_n])
        I = tf.eye(self.num_n,batch_shape=tf.expand_dims(batch_size,0))
        absrow = tf.tile(tf.expand_dims(I,2),[1,1,self.num_n,1])
        abscol = tf.tile(tf.expand_dims(I,1),[1,self.num_n,1,1])
        out_demand = tf.multiply(expand_demand, absrow)
        in_demand = tf.multiply(expand_demand,abscol)
        expand_avail_degree = tf.tile(tf.expand_dims(tf.expand_dims(available_degrees,1),1),[1,self.num_n,self.num_n,1])
        current_degrees = tf.reduce_sum(adj,axis=-1)
        expand_curr_degree = tf.tile(tf.expand_dims(tf.expand_dims(current_degrees,1),1),[1,self.num_n,self.num_n,1])
        features = [tf.expand_dims(expand_avail_degree,-1),tf.expand_dims(expand_curr_degree,-1),\
            tf.expand_dims(out_demand,-1),tf.expand_dims(in_demand,-1)]
        node_features = tf.concat(features,-1)
        return node_features

    def pad_neighbor_features(self, adj, node_features):
        """Padding neighbor features into MLP input dimension (max_degree)
        Assume that all degrees <= max_degree.
        
        Args:
            adj: Adjacency matrix of size [batch_size x N x N]
        Returns:
            neighbor_features: of size [batch_size x N x N x N x max_degree x dim]
        """
        #deg = tf.reduce_sum(adj,axis=-1)
        #batch_size = tf.shape(adj)[0]
        #expand_adj = tf.expand_dims(tf.expand_dims(adj,1),1)
        batch_neighbor_features_list = []
        for batch_no in range(self.batch_size):
            batch_adj = tf.gather(adj,batch_no,axis=0)
            batch_node_features = tf.gather(node_features,batch_no,axis=0) # [N,N,N,dim]
            single_neighbor_features_list = []
            for v in range(self.num_n):
                sliced_adj = tf.gather(batch_adj,v,axis=-2) #[N,]
                neighbor_inds = tf.squeeze(tf.where(sliced_adj>0),axis=-1) # of size [deg,]
                deg = tf.shape(neighbor_inds)[0]
                #if tf.shape(neighbor_inds)[0] <= self.max_degree:
                v_neighbor_feature = tf.gather(batch_node_features,neighbor_inds,axis=-2) # [N,N,deg,dim]
                pad_deg = tf.expand_dims(tf.expand_dims(self.max_degree - deg,0),0)
                paddings = tf.pad(pad_deg,tf.constant([[2,1],[1,0]]))
                v_neighbor_feature = tf.expand_dims(tf.pad(v_neighbor_feature,paddings),2) #[N,N,1,max_deg,dim]
                single_neighbor_features_list.append(v_neighbor_feature)
            single_neighbor_features = tf.expand_dims(tf.concat(single_neighbor_features_list,2),0) #[1,N,N,N,max_deg,dim]
            batch_neighbor_features_list.append(single_neighbor_features)
        batch_neighbor_features = tf.concat(batch_neighbor_features_list,0)
        return batch_neighbor_features

    def _aggregate(self, dims):
        # length: number of layers + 1
        aggregators = []
        for layer in range(len(dims)-1):
            dim_mult = 2 if self.concat and (layer != 0) else 1
            # aggregator at current layer
            if layer == len(dims) - 2:
                aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                        dropout=self.dropout, concat=self.concat)
            elif layer==0:
                aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                        dropout=0.0, concat=self.concat)
            else:
                aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1],
                        dropout=self.dropout, concat=self.concat)
            aggregators.append(aggregator)
            
        return aggregators

    def node_iterations(self, aggregator, features_tminus1, neighbor_features_tminus1):
        """
        Args:
            features_tminus1: of size [batch_size, N, N, N, dim_in]
            neighbor_features_tminus1: of size [batch_size, N, N, N, max_degree, dim_in]
        """
        batch_size = tf.shape(features_tminus1)[0]
        dim_in =tf.shape(features_tminus1)[-1]
        self_features = tf.reshape(features_tminus1,\
            [batch_size*self.num_n*self.num_n,self.num_n,dim_in])
        neigh_features = tf.reshape(neighbor_features_tminus1,\
            [batch_size*self.num_n*self.num_n,self.num_n,self.max_degree,dim_in])
        features_t_list = []
        # Iterates among nodes
        for v in range(self.num_n):
            self_fv = tf.gather(self_features,v,axis=1)
            neigh_fv = tf.gather(neigh_features,v,axis=1)
            fv_t = aggregator.call((self_fv,neigh_fv)) # [batch_size x N^2, dim_out * 2]
            fv_t = tf.expand_dims(fv_t,1)
            features_t_list.append(fv_t)
        features_t = tf.concat(features_t_list,1)
        features_t = tf.reshape(features_t, [batch_size,self.num_n,self.num_n,self.num_n,aggregator.output_dim*2])
        return features_t

    def demands_sumup(self, node_features):
        """
        Args:
            node_features: of size [batch_size, N, N, N, output_dim]
        """
        batch_size = tf.shape(node_features)[0]
        features = tf.reshape(node_features,[batch_size,self.num_n**2, self.num_n, 2*self.dims[-1]])
        features = tf.reduce_sum(features,axis=1)
        return features