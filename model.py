import tensorflow as tf
import math
import numpy as np


class Model:
    def __init__(self, vocab_dim, emb_dim):
        self._vocab_dim = vocab_dim
        self._emb_dim = emb_dim
        self._assemble_graph()

    def _assemble_graph(self):
        self._create_placeholders()

        A_var = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[self._vocab_dim, self._emb_dim],
                stddev=1 / math.sqrt(self._emb_dim)
            )
        )
        B_var = tf.Variable(
            initial_value=tf.truncated_normal(
                shape=[self._vocab_dim, self._emb_dim],
                stddev=1 / math.sqrt(self._emb_dim)
            )
        )

        resp_mult = tf.matmul(self.response_batch, B_var)
        cont_mult = tf.matmul(self.context_batch, A_var)

        self.f = tf.diag_part(tf.matmul(tf.transpose(cont_mult), resp_mult))
        tf.Print(self.f, [self.f])

        m = 0.01
        self.loss = tf.reduce_sum(tf.nn.relu(self.f_neg - self.f + m))

        LR = 0.001
        self.optimizer = tf.train.GradientDescentOptimizer(LR).minimize(self.loss)

    def _create_placeholders(self):
        self.context_batch = tf.placeholder(dtype=tf.float32, name='Context', shape=[None, self._vocab_dim])
        self.response_batch = tf.placeholder(dtype=tf.float32, name='Response', shape=[None, self._vocab_dim])
        self.f_neg = tf.placeholder(dtype=tf.float32, name='f_neg', shape=[None])

