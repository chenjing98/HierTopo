import numpy as np
import tensorflow as tf

class ActorAgent:
    def __init__(self, feature_num, edge_num, eps=1e-4, 
                 act_fn=tf.nn.leaky_relu, optimizer=tf.train.AdamOptimizer, scope='actor_agent'):
        #param
        self.sess = tf.Session()
        self.feature_num = feature_num
        self.node_num = edge_num
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope
        self.edge_num = edge_num #change!!!
        self.lr_rate = 1e-5
        self.hidden_dim_A = 6
        self.hidden_dim_B = 8
        self.hidden_dim_C = 10
        #many results
        self.node_inputs = tf.placeholder(tf.float32, shape=[None, self.edge_num, self.feature_num]) 
        self.entropy_weight = tf.placeholder(tf.float32, ())
        self.edge_act_vec = tf.placeholder(tf.float32, shape=[None, self.edge_num])
        self.weights_A = tf.Variable(tf.ones([self.feature_num,self.hidden_dim_A]))
        self.bias_A = tf.Variable(tf.ones([self.edge_num,self.hidden_dim_A]))
        self.weights_B = tf.Variable(tf.ones([self.hidden_dim_A,self.hidden_dim_B]))
        self.bias_B = tf.Variable(tf.ones([self.edge_num,self.hidden_dim_B]))
        self.weights_C = tf.Variable(tf.ones([self.hidden_dim_B,self.hidden_dim_C]))
        self.bias_C = tf.Variable(tf.ones([self.edge_num,self.hidden_dim_C]))
        self.adjm = tf.ones([self.edge_num,self.edge_num])#is constant??
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        
        #result
        self.edge_act_probs= self.actor_network(node_input=self.node_inputs, act_fn=self.act_fn)
        self.edge_acts = tf.argmax(self.edge_act_probs, 1)
        self.adv = tf.placeholder(tf.float32, [None, 1])
        self.selected_edge_prob = tf.reduce_sum(tf.multiply(self.edge_act_probs, self.edge_act_vec),
                                                reduction_indices=1, keep_dims=True)

        #loss
        self.adv_loss = tf.reduce_sum(tf.multiply(tf.log(self.selected_edge_prob + self.eps),
                                                  -self.adv))
        self.edge_entropy = tf.reduce_sum(tf.multiply(self.edge_act_probs, 
                                                      tf.log(self.edge_act_probs + self.eps)))
        self.entropy_loss = self.edge_entropy
        self.entropy_loss /= tf.log(tf.cast(tf.shape(self.edge_act_probs)[1], tf.float32))
        self.act_loss = self.adv_loss + self.entropy_weight * self.entropy_loss
        tf.summary.scalar('loss',self.act_loss)
        #gradient
        self.act_gradients = tf.gradients(self.act_loss, self.params)
        self.act_opt = self.optimizer(self.lr_rate).minimize(self.act_loss)

        #save
        self.sess.run(tf.global_variables_initializer())
        self.summary_ops = tf.summary.merge_all()
        self.writer_data = tf.summary.FileWriter('./data')

    def GCN_layer(self,node_input):
        #layer_A
        g = tf.transpose(node_input,[1,0,2])
        g = tf.matmul(self.adjm,tf.reshape(g,[self.edge_num,-1]))
        g = tf.reshape(g,[self.edge_num,-1,self.feature_num])
        g = tf.transpose(g,[1,0,2])
        g = tf.reshape(g,[-1,self.feature_num])
        g = tf.matmul(g,self.weights_A)
        g = tf.reshape(g,[-1,self.edge_num,self.hidden_dim_A])
        g = tf.add(g,self.bias_A)
        #layer_B
        g = tf.transpose(g,[1,0,2])
        g = tf.matmul(self.adjm,tf.reshape(g,[self.edge_num,-1]))
        g = tf.reshape(g,[self.edge_num,-1,self.hidden_dim_A])
        g = tf.transpose(g,[1,0,2])
        g = tf.reshape(g,[-1,self.hidden_dim_A])
        g = tf.matmul(g,self.weights_B)
        g = tf.reshape(g,[-1,self.edge_num,self.hidden_dim_B])
        g = tf.add(g,self.bias_B)
        #layer_C
        g = tf.transpose(g,[1,0,2])
        g = tf.matmul(self.adjm,tf.reshape(g,[self.edge_num,-1]))
        g = tf.reshape(g,[self.edge_num,-1,self.hidden_dim_B])
        g = tf.transpose(g,[1,0,2])
        g = tf.reshape(g,[-1,self.hidden_dim_B])
        g = tf.matmul(g,self.weights_C)
        node_output = tf.reshape(g,[-1,self.edge_num,self.hidden_dim_C])  
        node_output = tf.add(node_output,self.bias_C)

        return node_output    

    def actor_network(self,node_input,act_fn):
        node_feature = self.GCN_layer(node_input)
        node_feature = tf.reshape(node_feature,shape=[-1,self.edge_num*self.hidden_dim_C])
        edge_act_A = tf.contrib.layers.fully_connected(node_feature,32,activation_fn = tf.nn.relu)
        edge_act_B = tf.contrib.layers.fully_connected(edge_act_A,16,activation_fn = tf.nn.relu)
        edge_act_C = tf.contrib.layers.fully_connected(edge_act_B,8,activation_fn = tf.nn.relu)
        edge_act_probs = tf.contrib.layers.fully_connected(edge_act_C,self.edge_num,activation_fn = tf.nn.relu)
        edge_act = tf.nn.softmax(edge_act_probs)
        return edge_act_probs
        

    # node and edge here are referred to the same.
    def predict(self, node_inputs):
        return self.sess.run([self.edge_act_probs, self.edge_acts], feed_dict={
            self.node_inputs:node_inputs
        })

    
    def get_gradients(self, node_inputs, entropy_weight, adv, action):
        self.sess.run([self.act_gradients, self.act_loss, self.entropy_loss], feed_dict={
            self.node_inputs : node_inputs,
            self.entropy_weight : entropy_weight,
            self.adv : adv,
            self.edge_act_vec: action
        })   
    
    def train(self, node_inputs, entropy_weight, adv, action):
        _, summary_str = self.sess.run([self.act_opt,self.summary_ops], feed_dict={
            self.node_inputs : node_inputs,
            self.entropy_weight : entropy_weight,
            self.adv : adv,
            self.edge_act_vec: action
        })          
        self.writer_data.add_summary(summary_str)


class TopoEnv():

    def __init__(self, node_num, edge_num):

        self.penalty = 500
        self.degree_penalty = 5000
        self.edge_num = edge_num
        self.node_num = node_num
        #define demand and allowed degree
        self.demand = np.zeros((self.node_num,self.node_num))
        self.feature = np.zeros((self.edge_num,6))
        self.allowed_degree = np.ones((self.node_num))
        self.state = np.zeros((self.node_num,self.node_num))
        self.last_reward = 0
        self.count = 0

    def reset(self):
        for i in range(self.edge_num):
            n = self.edge_to_node(i)
            n1 = n[0]
            n2 = n[1]
            # other feature???
            self.feature[i][0] = self.demand[n1,n2] + self.demand[n2,n1]
            self.feature[i][1] = self.allowed_degree[n1]
            self.feature[i][2] = self.allowed_degree[n2]
            self.feature[i][3] = n1
            self.feature[i][4] = n2
            self.feature[i][5] = 0
        self.state = np.zeros((self.node_num,self.node_num))
        self.last_reward = 0
        self.demand = np.random.randint(1000, 2000, size=(self.node_num, self.node_num))
        for i in range(self.node_num):
            self.demand[i,i] = 0
        self.allowed_degree = [5,5,5,5,5,5,5,5]
        self.count = 0

    #i,j ----> min(i,j)
    def edge_to_node(self,e):
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                if ((i*(2*self.node_num-1-i)/2-1+j-i)== e):
                    return [i,j]

    def step(self,action):
        last = self.last_reward
        n = self.edge_to_node(action)
        n1 = n[0]
        n2 = n[1]
        if self.degree_check(action,n1,n2):
            if self.valid_check(action):
                self.feature[action][5] = 1
                self.state[n1,n2] = 1
                self.state[n2,n1] = 1
                self.degree_decline(n1,n2)
                reward = self.get_reward()
            else:
                reward = -self.penalty
                self.last_reward = reward
        else:
            reward = -self.degree_penalty
            self.last_reward = reward
        if reward < 0:
            self.count += 1
        if (self.count >= self.node_num):
            stop = True
        else:
            stop = False
        return reward,self.feature,stop
    
    def check_full(self):
        check_one = 1
        for i in range(self.node_num):
            check_two = 0
            for j in range(self.node_num):
                if self.state[i,j] == 1:
                    check_two = 1
            if(check_two == 0):
                check_one = 0
        return check_one

    def degree_check(self,action,n1,n2):
        minf = 1
        maxf = 1
        for i in range(self.edge_num):
            if self.feature[i][3] == n1 or self.feature[i][3] == n2:
                if self.feature[i][1] <= 0:
                    minf = 0
        for i in range(self.edge_num):
            if self.feature[i][4] == n1 or self.feature[i][4] == n2:
                if self.feature[i][2] <= 0:
                    maxf = 0
        if (minf == 1 and maxf == 1):
            return True
        else:
            return False
    
    def degree_decline(self,n1,n2):
        for i in range(self.edge_num):
            if self.feature[i][3] == n1 or self.feature[i][3] == n2:
                self.feature[i][1] -= 1
        for i in range(self.edge_num):
            if self.feature[i][4] == n1 or self.feature[i][4] == n2:
                self.feature[i][2] -= 1

    def valid_check(self,action):
        if (self.feature[action][5] == 1):
            return False
        else:
            return True
        
    def get_reward(self):
        D = self.state.copy()

        for i in range(self.node_num):
            for j in range(self.node_num):
                if (D[i][j] == 0) & (i != j):
                    D[i][j] = 999

        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if(D[i][j]>D[i][k]+D[k][j]):
                        D[i][j]=D[i][k]+D[k][j]
        score = 0
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                if(D[i,j]) > self.node_num:
                    score += self.demand[i,j]*self.node_num
                else:
                    score += self.demand[i,j]*D[i,j]
        score = -score
        score = score/1000
        reward = score - self.last_reward
        self.last_reward = score
        return reward

    def baseline(self,count):
        demand = []
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                demand.append(self.demand[i,j])
        edge = []
        D = np.zeros((self.node_num,self.node_num))
        for k in range(count):
            e = demand.index(max(demand))
            edge.append(e)
            n = self.edge_to_node(e)
            n1 = n[0]
            n2 = n[1]
            D[n1,n2] = 1
            D[n2,n1] = 1
            demand[e] = -1000
        for i in range(self.node_num):
            for j in range(self.node_num):
                if (D[i][j] == 0) & (i != j):
                    D[i][j] = 999
        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if(D[i][j]>D[i][k]+D[k][j]):
                        D[i][j]=D[i][k]+D[k][j]
        score = 0
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                if(D[i,j]) > self.node_num:
                    score += self.demand[i,j]*self.node_num
                else:
                    score += D[i,j]*self.demand[i,j]
        
        for i in range(self.node_num):
            if np.sum(self.state[i,:]) > self.allowed_degree[i]:
                score = score - self.degree_penalty    
        score = - score
        score = score/1000
        return score   

    def matching(self,count):
        demand = []
        allowed_degree = self.allowed_degree
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                demand.append(self.demand[i,j])
        edge = []
        D = np.zeros((self.node_num,self.node_num))    
        c = 0
        while c < count:
            e = demand.index(max(demand))
            edge.append(e)
            n = self.edge_to_node(e)
            n1 = n[0]
            n2 = n[1]
            if allowed_degree[n1] > 0 and allowed_degree[n2] > 0:
                D[n1,n2] = 1
                D[n2,n1] = 1
                allowed_degree[n1] -= 1
                allowed_degree[n2] -= 1
                c += 1
            else:
                error += 1
            demand[e] = -1000 

        for i in range(self.node_num):
            for j in range(self.node_num):
                if (D[i][j] == 0) & (i != j):
                    D[i][j] = 999
        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if(D[i][j]>D[i][k]+D[k][j]):
                        D[i][j]=D[i][k]+D[k][j]
        score = 0
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                if(D[i,j]) > self.node_num:
                    score += self.demand[i,j]*self.node_num
                else:
                    score += D[i,j]*self.demand[i,j]          
        score = - score
        score = score/1000
        return score    

    def compute_value(self):
        D = self.state.copy()

        for i in range(self.node_num):
            for j in range(self.node_num):
                if (D[i][j] == 0) & (i != j):
                    D[i][j] = 999

        for k in range(self.node_num):
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if(D[i][j]>D[i][k]+D[k][j]):
                        D[i][j]=D[i][k]+D[k][j]
        score = 0
        for i in range(self.node_num-1):
            for j in range(i+1,self.node_num):
                if(D[i,j]) > self.node_num:
                    score += self.demand[i,j]*self.node_num
                else:
                    score += self.demand[i,j]*D[i,j]
        
        for i in range(self.node_num):
            if np.sum(self.state[i,:]) > self.allowed_degree[i]:
                score = score - self.degree_penalty
        score = - score
        score = score/1000
        return score

    def observe(self):
        return self.feature

if __name__ == "__main__":
    env = TopoEnv(8,28)
    actor_agent = ActorAgent(feature_num = 6, edge_num = 28)

    test_step = 0
    while test_step<100:
        print(test_step)

        env.reset()
        
        vec = np.empty((0,28))
        adv = np.empty((0,1))
        node_inputs = np.empty((0,28,6))

        feature = env.observe()
        stop = False
        #f=open('train.txt','a+')
        #exp = {'feature':[],'action': [], 'reward': [], 'batch_point':[]}
        count = 0 
        while not stop:
            t_feature = np.expand_dims(feature,axis = 0)
            probs,select = actor_agent.predict(t_feature)
            node_inputs = np.append(node_inputs,np.expand_dims(feature,axis =0),axis =0)
            action = np.zeros((1,28))
            action[0,select[0]] = 1
            vec = np.append(vec,action,axis =0)
            reward,feature,stop = env.step(select[0])
            print(reward)
            adv = np.append(adv,np.expand_dims(np.array([reward]),axis =0),axis = 0)
            actor_agent.train(node_inputs,1,adv,vec)
            count  += 1
        v1 = env.compute_value()
        v2 = env.baseline(count)
        v3 = env.matching(count)
        rev = v1 - v2
        for i in range(count):
            adv[i,0] = rev
        actor_agent.train(node_inputs,1,adv,vec)
        #f.write(str(test_step))
        #f.write('\n')
        #f.write(str(v1))
        #f.write('\n')
        #f.write(str(v2))
        #f.write('\n')
        #f.close()
        test_step += 1
