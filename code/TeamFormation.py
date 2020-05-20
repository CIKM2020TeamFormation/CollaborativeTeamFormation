import tensorflow as tf
import numpy as np

import sys

class teamformation:    
    def __init__(self,nbins,bacthsize,embeding_size,data):
        self.dataset=data
        pfile=open(self.dataset+"krnm_pro.txt")
        pfile.readline()
        properties=pfile.readline().strip().split(" ")
        pfile.close()
        self.max_q_len=int(properties[0]) # number of nodes in the CQA network graph N=|Qestions|+|Answers|+|Experts|                
        self.max_d_len=int(properties[1])
        self.vocab_size=int(properties[2])
        
        self.n_bins=nbins
        self.embedding_size=embeding_size
        self.batch_size=bacthsize
        self.lamb = 0.5

        self.mus = teamformation.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = teamformation.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        self.W = teamformation.weight_variable((self.n_bins, 1))
        self.b = tf.Variable(0.,dtype=tf.float32)
        
        self.embeddings = tf.Variable(tf.random.uniform([self.vocab_size+1, self.embedding_size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
        
        self.inputs_q=tf.Variable(( self.batch_size,self.max_q_len),dtype=tf.int32)
        self.inputs_posd=tf.Variable(( self.batch_size,self.max_d_len),dtype=tf.int32)
        self.inputs_negd=tf.Variable(( self.batch_size,self.max_d_len),dtype=tf.int32)
        
        
    #copied from knrm paper ref:https://github.com/AdeDZY/K-NRM
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

    #copied from knrm paper copied from knrm paper ref:https://github.com/AdeDZY/K-NRM
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
    #copied from knrm paper copied from knrm paper ref:https://github.com/AdeDZY/K-NRM
    def weight_variable(shape):
        tmp = np.sqrt(6.0) / np.sqrt(shape[0] + shape[1])
        initial = tf.random.uniform(shape, minval=-tmp, maxval=tmp)
        return tf.Variable(initial,dtype=tf.float32)    
    #addopted from knrm paper copied from knrm paper ref:https://github.com/AdeDZY/K-NRM
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
        #print(kde,self.q_weights)
        aggregated_kde = tf.reduce_sum(kde*self.q_weights , [1])  # [batch, n_bins]
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
        y=self.d_score/tf.reduce_sum(self.d_score,0)
        predicted_d_score_temp=tf.reshape(predicted_d_score,(self.batch_size,))        
        out=tf.nn.softmax(tf.reshape(predicted_d_score_temp,(self.batch_size,)))   
        logout=tf.math.log(out)
        y=tf.dtypes.cast(y, tf.float32)        
        mul=y * logout
        loss=tf.reduce_mean(-tf.reduce_sum(mul))
        return loss
              
    def read_train(self):
        #load data
            trainfile = open(self.dataset+"train.txt")
            line=trainfile.readline().strip()
            self.query=[]
            self.doc=[]
            self.docscore=[]
            while line:
                parts=line.split(",") 
                if len(parts)<2:
                    print (line)   
                    line=trainfile.readline().strip()
                    continue
                qw1=np.array([ int(t) for t in parts[0].strip().split(' ')])
                qw=np.ones(self.max_q_len,dtype=int) * 0 #-1
                qw[:len(qw1)]=qw1
                qlist=[]
                dlist=[]
                sco=[]
                for i in range(len(parts)-1):
                    d1=parts[i+1].strip().split(' ')
                    sc=float(d1[-1])                   
                    d1=np.array([ int(t) for t in parts[i+1].strip().split(' ')[:-1]])                   
                    if len(d1)>0:
                        d=np.ones(self.max_d_len,dtype=int)*0 #* -1
                        d[:len(d1)]=d1
                        dlist.append(d)
                        qlist.append(qw)
                        sco.append(sc) 
                if  len(dlist)>0:  
                    self.query.append(qlist) 
                    self.doc.append(dlist)
                    sco=np.array(sco)
                    self.docscore.append(sco)  
                line=trainfile.readline().strip()   
            
            trainfile.close()
    
    
    def read_eval(self):
            valfile = open(self.dataset+"validation.txt")
            line=valfile.readline().strip()
            self.valquery=[]
            self.valdoc=[]
            self.valdocscore=[]
            while line:
                parts=line.split(",") 
                if len(parts)<2:
                    print (line)   
                    line=valfile.readline().strip()
                    continue
                qw1=np.array([ int(t) for t in parts[0].strip().split(' ')])
                qw=np.ones(self.max_q_len,dtype=int) * 0#-1
                qw[:len(qw1)]=qw1
                qlist=[]
                dlist=[]
                sco=[]
                for i in range(len(parts)-1):
                    d1=parts[i+1].strip().split(' ')
                    sc=float(d1[-1])                   
                    d1=np.array([ int(t) for t in parts[i+1].strip().split(' ')[:-1]])                   
                    if len(d1)>0:
                        d=np.ones(self.max_d_len,dtype=int) * 0#-1
                        d[:len(d1)]=d1
                        dlist.append(d)
                        qlist.append(qw)
                        sco.append(sc) 
                if  len(dlist)>0:  
                    self.valquery.append(qlist) 
                    self.valdoc.append(dlist)
                    sco=np.array(sco)
                    self.valdocscore.append(sco)  
                line=valfile.readline().strip()  
            valfile.close()
         
    def save_model(self):
        np.savetxt(self.dataset+'model/embeddings.txt',self.embeddings.numpy())
        np.savetxt(self.dataset+'model/b.txt',[self.b.numpy()])
        np.savetxt(self.dataset+'model/W.txt',self.W.numpy())
        
        
    def train(self):   
        self.read_train()
        self.read_eval()
        
        epochs = range(6)
        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        t_loss=0
        for epoch in epochs: 
            for i in range(len(self.query)):
                self.inputs_q=self.query[i]
                self.inputs_posd=self.doc[i]
                self.batch_size=len(self.doc[i]) 
                self.d_score=self.docscore[i]  
                self.q_weights=np.where(np.array(self.inputs_q)>0,1,0)
                self.q_weights=tf.dtypes.cast(self.q_weights, tf.float32)
                self.q_weights = tf.reshape(self.q_weights, shape=[self.batch_size, self.max_q_len, 1])
                #print(self.q_weights)
                opt.minimize(ob.loss, var_list=[ob.W,ob.b,ob.embeddings])
                t_loss+=ob.loss()
                if i%500==0:
                    tf.print("epoch:%d, i:%d, loss:%f"%(epoch,i,t_loss/(epoch*len(self.query)+i+1)))
            if epoch%1==0:
                tf.print("epoch:%d,  loss:%f"%(epoch,t_loss/((epoch+1)*len(self.query))))
                    
            val_loss=0        
            for i in range(len(self.valquery)):
                self.inputs_q=self.valquery[i]
                self.inputs_posd=self.valdoc[i]
                self.batch_size=len(self.valdoc[i]) 
                self.d_score=self.valdocscore[i]  
                self.q_weights=np.where(np.array(self.inputs_q)>0,1,0)
                self.q_weights=tf.dtypes.cast(self.q_weights, tf.float32)
                self.q_weights = tf.reshape(self.q_weights, shape=[self.batch_size, self.max_q_len, 1])
                opt.minimize(ob.loss, var_list=[ob.W,ob.b,ob.embeddings])
                val_loss+=ob.loss()
            if epoch%1==0:
                tf.print("epoch:%d,  validation loss:%f"%(epoch,val_loss/(len(self.valquery))))        
            self.save_model()        

        
dataset=["../data/apple/","../data/dba/","../data/electronics/","../data/history/","../data/english/","../data/softwareengineering/"]        
ob=teamformation(nbins=11,bacthsize=16,embeding_size=32,data=dataset[1])        
ob.train()
