import tensorflow as tf
import numpy as np


class teamformation:    
    def __init__(self,nbins,bacthsize):
        self.n_bins=nbins
        self.embedding_size=2
        self.batch_size=bacthsize
        self.max_q_len=2
        self.max_d_len=4
        self.lamb = 0.5
        self.mus = com2vec.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = com2vec.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        self.W = com2vec.weight_variable((self.n_bins, 1))
        self.b = tf.Variable(0.,dtype=tf.float32)
        
        self.embeddings = tf.Variable(tf.random.uniform([self.N , self.embedding_size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
        
        self.inputs_q=tf.Variable(( self.bachsize,self.max_q_len),dtype=tf.int32)
        self.inputs_posd=tf.Variable(( self.bachsize,self.max_d_len),dtype=tf.int32)
        self.inputs_negd=tf.Variable(( self.bachsize,self.max_d_len),dtype=tf.int32)
        
        
    #copied from K-NRM package https://github.com/AdeDZY/K-NRM/blob/master/LICENSE
    @staticmethod
    def kernal_mus(n_kernels, use_exact):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        if use_exact:
            l_mu = [1]
        else:
            l_mu = [2]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    #copied from K-NRM package https://github.com/AdeDZY/K-NRM/blob/master/LICENSE
    @staticmethod
    def kernel_sigmas(n_kernels, lamb, use_exact):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.00001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [bin_size * lamb] * (n_kernels - 1)
        return l_sigma
    #copied from K-NRM package https://github.com/AdeDZY/K-NRM/blob/master/LICENSE
    def weight_variable(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial,dtype=tf.float32)    

    def model(self,inputs_q,inputs_d):    
        # look up embeddings for each term. [nbatch, qlen, emb_dim]
        q_embed = tf.nn.embedding_lookup(self.embeddings, inputs_q, name='qemb')
        d_embed = tf.nn.embedding_lookup(self.embeddings, inputs_d, name='demb')
        #print(q_embed)
        #print(d_embed)
        ## Uingram Model
        # normalize and compute similarity matrix using l2 norm         
        norm_q = tf.sqrt(tf.reduce_sum(tf.square(q_embed), 2))
        #print(norm_q)
        norm_q=tf.reshape(norm_q,(len(norm_q),len(norm_q[0]),1))
        #print(norm_q)
        normalized_q_embed = q_embed / norm_q
        #print(normalized_q_embed)
        norm_d = tf.sqrt(tf.reduce_sum(tf.square(d_embed), 2))
        norm_d=tf.reshape(norm_d,(len(norm_d),len(norm_d[0]),1))
        normalized_d_embed = d_embed / norm_d
        #print(normalized_d_embed)
        tmp = tf.transpose(normalized_d_embed, perm=[0, 2, 1])
        #print(tmp)
        sim =tf.matmul(normalized_q_embed, tmp)
        #print(sim)        
        # compute gaussian kernel
        rs_sim = tf.reshape(sim, [self.batch_size, self.max_q_len, self.max_d_len, 1])
        #print(rs_sim)
        
        tmp = tf.exp(-tf.square(tf.subtract(rs_sim, self.mus)) / (tf.multiply(tf.square(self.sigmas), 2)))
        #print(tmp)
        
        feats = []  # store the soft-TF features from each field.
        # sum up gaussian scores
        kde = tf.reduce_sum(tmp, [2])
        kde = tf.math.log(tf.maximum(kde, 1e-10)) * 0.01  # 0.01 used to scale down the data.
        # [batch_size, qlen, n_bins]
        
        #print(kde)
        # aggregated query terms
        # q_weights = [1, 1, 0, 0...]. Works as a query word mask.
        # Support query-term weigting if set to continous values (e.g. IDF).
        aggregated_kde = tf.reduce_sum(kde , [1])  # [batch, n_bins]
        #print( aggregated_kde)
        feats.append(aggregated_kde) # [[batch, nbins]]
        feats_tmp = tf.concat( feats,1)  # [batch, n_bins]
        #print ("batch feature shape:", feats_tmp.get_shape())
        
        # Reshape. (maybe not necessary...)
        feats_flat = tf.reshape(feats_tmp, [-1, self.n_bins])
        #print ("flat feature shape:", feats_flat.get_shape())
        
        # Learning-To-Rank layer. o is the final matching score.
        o = tf.tanh(tf.matmul(feats_flat, self.W) + self.b)
        #print(o)
        return o
    
    def loss(self):
        predicted_d_score=self.model(self.inputs_q,self.inputs_posd)
        y=self.d_score/tf.reduce_sum(d_score,0)
        predicted_d_score_temp=tf.reshape(predicted_d_score,(batch_size,))        
        out=tf.nn.softmax(tf.reshape(predicted_d_score_temp,(batch_size,)))       
        loss=tf.reduce_mean(-tf.reduce_sum(y * tf.log(out)))
        return loss
              
    def run():            
        epochs = range(400)
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        for epoch in epochs: 
            for i in range(int(tarin_size/self.batch_size)):
            self.inputs_q,self.inputs_d, self.score_d=self.read_train(filestream,self.batch_size)
            opt.minimize(ob.loss, var_list=[ob.W,ob.b,ob.embeddings])
            if epoch%100==0:
                tf.print(ob.loss())
                
        
        
teamformation.run()
