import tensorflow as tf
import numpy as np
from models.attn_head import sp_attn_head
from Parameters import Parameters as pm


class GAT(object):
    def __init__(self):
        print('start building model...')
        self.input_x = tf.placeholder(dtype=tf.int32, shape=(None, pm.seq_length), name='input_x')
        self.input_mask = tf.placeholder(dtype=tf.float32, shape=(None, pm.seq_length), name='input_mask')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=(None,pm.num_classes), name='input_y')
        self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(1, pm.nb_nodes, pm.ft_size), name='ftr_in')
        self.bias_in = tf.sparse_placeholder(dtype=tf.float32)
        self.attn_drop = tf.placeholder(dtype=tf.float32, name='attn_drop')
        self.ffd_drop = tf.placeholder(dtype=tf.float32, name='ffd_drop')
        self.inference()
        print('building model finished!')

    def inference(self):
        # 输入包括：1.inputs (batch_size,nb_mentions) 2.nodes_features (1,nb_nodes,nb_features)整个网络的节点特征 3.bias_mat 整个网络的结构
        # 第1层GAT (1,nb_nodes,nb_features) -> (1,nb_nodes,256) -> (1,nb_nodes,256*8)
        with tf.name_scope('GAT_1'):
            attns = []
            for _ in range(pm.n_heads[0]):
                attns.append(
                    sp_attn_head(self.ftr_in, adj_mat=self.bias_in, out_sz=pm.hid_units[0], activation=pm.nonlinear[0],
                                 nb_nodes=pm.nb_nodes, in_drop=pm.ffd_drop, coef_drop=pm.attn_drop,
                                 residual=pm.residual))
            h_1 = tf.concat(attns, axis=-1)

        # 第2层GAT (1,nb_nodes,256*8) -> (1,nb_nodes,128) -> (1,nb_nodes,128*4)
        with tf.name_scope('GAT_2'):
            attns = []
            for _ in range(pm.n_heads[-1]):
                attns.append(
                    sp_attn_head(h_1, adj_mat=self.bias_in, out_sz=pm.hid_units[-1], activation=pm.nonlinear[-1],
                                 nb_nodes=pm.nb_nodes, in_drop=pm.ffd_drop, coef_drop=pm.attn_drop,
                                 residual=pm.residual))
            h_2 = tf.concat(attns, axis=-1)

        # attention
        with tf.name_scope('attention'):
            attention_w = tf.Variable(
                tf.truncated_normal([pm.hid_units[-1] * pm.n_heads[-1], pm.attention_size], stddev=0.1),
                name='attention_w')
            attention_u = tf.Variable(tf.truncated_normal([pm.attention_size, 1], stddev=0.1), name='attention_u')
            attention_b = tf.Variable(tf.constant(0.1, shape=[pm.attention_size]), name='attention_b')

            u_list = []
            embedding_input = tf.nn.embedding_lookup(tf.squeeze(h_2), self.input_x)  # (batch_size,nb_mentions,128*4)
            for t in range(pm.seq_length):
                u_t = tf.matmul(embedding_input[:, t, :], attention_w) + tf.reshape(attention_b, [1,
                                                                                                  -1])  # (batch_size,128*4) (128*4,attention_sizse) -> (batch_size,attention_size)
                u = tf.matmul(u_t, attention_u)  # (batch_size,attention_size) (attention_size,1) -> (batch_size,1)
                u_list.append(u)
            logit = tf.concat(u_list, axis=1)  # (batch_size,seq_len)=(batch_size,nb_mentions)

            logit = logit * self.input_mask + (1 - self.input_mask) * (-1e9)

            weights = tf.nn.softmax(logit, name='attention_weights')
            out1 = tf.reduce_sum(embedding_input * tf.reshape(weights, [-1, pm.seq_length, 1]),
                                 1)  # (batch_size,nb_mentions,128*4) (batch_size,nb_mentions,1) -> (batch_size,128*4)

        # MLP
        with tf.name_scope('output'):
            weight = tf.Variable(tf.truncated_normal([pm.hid_units[-1] * pm.n_heads[-1], pm.num_classes], stddev=0.1),
                                 name='weight')
            bias = tf.Variable(tf.constant(0.1, shape=[pm.num_classes]), name='bias')

            self.logits = tf.matmul(out1, weight) + tf.reshape(bias, [1, -1])  # (batch_size,num_classes)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)  # (batch)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            self.train_op = optimizer.minimize(self.loss)

    def feed_data(self, batch_x, batch_y, batch_mask, features, biases, attn_drop, ffd_drop):
        feed_dict = {self.input_x: batch_x,
                     self.input_y: batch_y,
                     self.input_mask: batch_mask,
                     self.ftr_in: features,
                     self.bias_in: biases,
                     self.attn_drop: attn_drop,
                     self.ffd_drop: ffd_drop
                     }
        return feed_dict

