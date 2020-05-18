import tensorflow as tf
import numpy as np


class TeamFormation:    
    def __init__(self,n,m,nbins):
        self.N=n
        self.M=m
        self.n_bins=nbins
        self.embedding_size=2
        self.batch_size=3
        self.max_q_len=2
        self.max_d_len=4
        self.lamb = 0.5
        self.mus = com2vec.kernal_mus(self.n_bins, use_exact=True)
        self.sigmas = com2vec.kernel_sigmas(self.n_bins, self.lamb, use_exact=True)
        self.W = com2vec.weight_variable((self.n_bins, 1))
        self.b = tf.Variable(0.,dtype=tf.float32)
        
        self.embeddings = tf.Variable(tf.random.uniform([self.N , self.embedding_size], -1.0, 1.0,dtype=tf.float32),dtype=tf.float32)
        
        self.inputs_q=tf.Variable((3,self.max_q_len),dtype=tf.int32)
        self.inputs_posd=tf.Variable((3,self.max_d_len),dtype=tf.int32)
        self.inputs_negd=tf.Variable((3,self.max_d_len),dtype=tf.int32)
    
  