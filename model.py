import tensorflow as tf
import math
import numpy as np


class Model:
    def __init__(self, vocab_dim, emb_dim):
        self._vocab_dim = vocab_dim
        self._emb_dim = emb_dim
        self._assemble_graph()
        tf.set_random_seed(42)

    def _assemble_graph(self):
        self._create_placeholders()

        A_var = tf.Variable(
            initial_value=tf.random_uniform(
                shape=[self._emb_dim, self._vocab_dim],
                minval=-1, maxval=1
            )
        )
        B_var = tf.Variable(
            initial_value=tf.random_uniform(
                shape=[self._emb_dim, self._vocab_dim],
                minval=-1, maxval=1
            )
        )

        cont_mult = tf.transpose(tf.matmul(A_var, tf.transpose(self.context_batch)))
        resp_mult = tf.matmul(B_var, tf.transpose(self.response_batch))
        neg_resp_mult = tf.matmul(B_var, tf.transpose(self.neg_response_batch))

        pos_raw_f = tf.diag_part(tf.matmul(cont_mult, resp_mult))
        neg_raw_f = tf.diag_part(tf.matmul(cont_mult, neg_resp_mult))
        self.f_pos = pos_raw_f
        self.f_neg = neg_raw_f

        m = 0.01
        self.loss = tf.reduce_sum(tf.nn.relu(self.f_neg - self.f_pos + m))

        LR = 0.0001
        self.optimizer = tf.train.GradientDescentOptimizer(LR).minimize(self.loss)

    def _create_placeholders(self):
        self.context_batch = tf.placeholder(dtype=tf.float32, name='Context', shape=[None, self._vocab_dim])
        self.response_batch = tf.placeholder(dtype=tf.float32, name='Response', shape=[None, self._vocab_dim])
        self.neg_response_batch = tf.placeholder(dtype=tf.float32, name='NegResponse', shape=[None, self._vocab_dim])
