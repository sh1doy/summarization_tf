"""layers"""

import tensorflow as tf
from utils import *
tfe = tf.contrib.eager


class TreeEmbeddingLayer(tf.keras.Model):
    def __init__(self, dim_E, in_vocab):
        super(TreeEmbeddingLayer, self).__init__()
        self.E = tf.get_variable("E", [in_vocab, dim_E], tf.float32,
                                 initializer=tf.keras.initializers.RandomUniform())

    def call(self, x):
        '''x: list of [1,]'''
        x_len = [xx.shape[0] for xx in x]
        ex = tf.nn.embedding_lookup(self.E, tf.concat(x, axis=0))
        exs = tf.split(ex, x_len, 0)
        return exs


class TreeEmbeddingLayerTreeBase(tf.keras.Model):
    def __init__(self, dim_E, in_vocab):
        super(TreeEmbeddingLayerTreeBase, self).__init__()
        self.E = tf.get_variable("E", [in_vocab, dim_E], tf.float32,
                                 initializer=tf.keras.initializers.RandomUniform())

    def call(self, roots):
        return [self.apply_single(root) for root in roots]

    def apply_single(self, root):
        labels = traverse_label(root)
        embedded = tf.nn.embedding_lookup(self.E, labels)
        new_nodes = self.Node2TreeLSTMNode(root, parent=None)
        for rep, node in zip(embedded, traverse(new_nodes)):
            node.h = rep
        return new_nodes

    def Node2TreeLSTMNode(self, node, parent):
        children = [self.Node2TreeLSTMNode(c, node) for c in node.children]
        return TreeLSTMNode(node.label, parent=parent, children=children, num=node.num)


class ChildSumLSTMLayerWithEmbedding(tf.keras.Model):
    def __init__(self, in_vocab, dim_in, dim_out):
        super(ChildSumLSTMLayerWithEmbedding, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.E = tf.get_variable("E", [in_vocab, dim_in], tf.float32,
                                 initializer=tf.keras.initializers.RandomUniform())
        self.U_f = tf.keras.layers.Dense(dim_out, use_bias=False)
        self.U_iuo = tf.keras.layers.Dense(dim_out * 3, use_bias=False)
        self.W = tf.keras.layers.Dense(dim_out * 4)
        # self.h_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        # self.c_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.h_init = tf.zeros([1, dim_out], tf.float32)
        self.c_init = tf.zeros([1, dim_out], tf.float32)

    @staticmethod
    def get_nums(roots):
        res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
        max_len = max([len(x) for x in res])
        res = tf.keras.preprocessing.sequence.pad_sequences(
            res, max_len, padding="post", value=-1.)
        return tf.constant(res, tf.int32)

    def call(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch2(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        indices = [self.get_nums(nodes) for nodes in depthes]

        h_tensor = self.h_init
        c_tensor = self.c_init
        for indice, nodes in zip(indices, depthes):
            x = tf.nn.embedding_lookup(self.E, [node.label for node in nodes])  # [nodes, dim_in]
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice, nodes)
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
        return depthes[-1]

    def apply(self, x, h_tensor, c_tensor, indice, nodes):

        mask_bool = tf.not_equal(indice, -1.)
        mask = tf.cast(mask_bool, tf.float32)  # [batch, child]

        h = tf.gather(h_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))  # [nodes, child, dim]
        c = tf.gather(c_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))
        h_sum = tf.reduce_sum(h * tf.expand_dims(mask, -1), 1)  # [nodes, dim_out]

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = tf.reshape(self.U_f(tf.reshape(h, [-1, h.shape[-1]])), h.shape)
        branch_f_k = tf.sigmoid(tf.expand_dims(W_f_x, 1) + branch_f_k)
        branch_f = tf.reduce_sum(branch_f_k * c * tf.expand_dims(mask, -1), 1)  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = tf.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = tf.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = tf.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)  # [node, dim_out]

        for n, c, h in zip(nodes, new_c, new_h):
            n.c = c
            n.h = h

        return new_h, new_c


class ChildSumLSTMLayer(tf.keras.Model):
    def __init__(self, dim_in, dim_out):
        super(ChildSumLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = tf.keras.layers.Dense(dim_out, use_bias=False)
        self.U_iuo = tf.keras.layers.Dense(dim_out * 3, use_bias=False)
        self.W = tf.keras.layers.Dense(dim_out * 4)
        # self.h_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        # self.c_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.h_init = tf.zeros([1, dim_out], tf.float32)
        self.c_init = tf.zeros([1, dim_out], tf.float32)

    def call(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice)
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
            res_h.append(h_tensor[1:, :])
            res_c.append(c_tensor[1:, :])
        return res_h, res_c

    def apply(self, x, h_tensor, c_tensor, indice):

        mask_bool = tf.not_equal(indice, -1.)
        mask = tf.cast(mask_bool, tf.float32)  # [batch, child]

        h = tf.gather(h_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))  # [nodes, child, dim]
        c = tf.gather(c_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))
        h_sum = tf.reduce_sum(h * tf.expand_dims(mask, -1), 1)  # [nodes, dim_out]

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = tf.reshape(self.U_f(tf.reshape(h, [-1, h.shape[-1]])), h.shape)
        branch_f_k = tf.sigmoid(tf.expand_dims(W_f_x, 1) + branch_f_k)
        branch_f = tf.reduce_sum(branch_f_k * c * tf.expand_dims(mask, -1), 1)  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = tf.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = tf.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = tf.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)  # [node, dim_out]

        return new_h, new_c


class ChildSumLSTMLayerTreeBase(tf.keras.Model):
    def __init__(self, dim_in, dim_out):
        super(ChildSumLSTMLayerTreeBase, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = tf.keras.layers.Dense(dim_out, use_bias=False)
        self.U_iuo = tf.keras.layers.Dense(dim_out * 3, use_bias=False)
        self.W = tf.keras.layers.Dense(dim_out * 4)
        # self.h_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        # self.c_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.h_init = tf.zeros([1, dim_out], tf.float32)
        self.c_init = tf.zeros([1, dim_out], tf.float32)

    @staticmethod
    def get_nums(roots):
        res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
        max_len = max([len(x) for x in res])
        res = tf.keras.preprocessing.sequence.pad_sequences(
            res, max_len, padding="post", value=-1.)
        return tf.constant(res, tf.int32)

    def call(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch2(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        indices = [self.get_nums(nodes) for nodes in depthes]

        h_tensor = self.h_init
        c_tensor = self.c_init
        for indice, nodes in zip(indices, depthes):
            x = tf.stack([node.h for node in nodes])  # [nodes, dim_in]
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice, nodes)
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
        return depthes[-1]

    def apply(self, x, h_tensor, c_tensor, indice, nodes):

        mask_bool = tf.not_equal(indice, -1.)
        mask = tf.cast(mask_bool, tf.float32)  # [batch, child]

        h = tf.gather(h_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))  # [nodes, child, dim]
        c = tf.gather(c_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))
        h_sum = tf.reduce_sum(h * tf.expand_dims(mask, -1), 1)  # [nodes, dim_out]

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = tf.reshape(self.U_f(tf.reshape(h, [-1, h.shape[-1]])), h.shape)
        branch_f_k = tf.sigmoid(tf.expand_dims(W_f_x, 1) + branch_f_k)
        branch_f = tf.reduce_sum(branch_f_k * c * tf.expand_dims(mask, -1), 1)  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = tf.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = tf.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = tf.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)  # [node, dim_out]

        for n, c, h in zip(nodes, new_c, new_h):
            n.c = c
            n.h = h

        return new_h, new_c


class NaryLSTMLayer(tf.keras.Model):
    def __init__(self, dim_in, dim_out):
        super(NaryLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f1 = tf.keras.layers.Dense(dim_out, use_bias=False)
        self.U_f2 = tf.keras.layers.Dense(dim_out, use_bias=False)
        self.U_iuo = tf.keras.layers.Dense(dim_out * 3, use_bias=False)
        self.W = tf.keras.layers.Dense(dim_out * 4)
        # self.h_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        # self.c_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.h_init = tf.zeros([1, dim_out], tf.float32)
        self.c_init = tf.zeros([1, dim_out], tf.float32)

    def call(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice)
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
            res_h.append(h_tensor[1:, :])
            res_c.append(c_tensor[1:, :])
        return res_h, res_c

    def apply(self, x, h_tensor, c_tensor, indice):

        mask_bool = tf.not_equal(indice, -1.)

        h = tf.gather(h_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))  # [nodes, child, dim]
        c = tf.gather(c_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        if h.shape[1] <= 1:
            h = tf.concat([h, tf.zeros_like(h)], 1)  # [nodes, 2, dim]
            c = tf.concat([c, tf.zeros_like(c)], 1)

        h_concat = tf.reshape(h, [h.shape[0], -1])

        branch_f1 = self.U_f1(h_concat)
        branch_f1 = tf.sigmoid(W_f_x + branch_f1)
        branch_f2 = self.U_f2(h_concat)
        branch_f2 = tf.sigmoid(W_f_x + branch_f2)
        branch_f = branch_f1 * c[:, 0] + branch_f2 * c[:, 1]

        branch_iuo = self.U_iuo(h_concat)  # [nodes, dim_out * 3]
        branch_i = tf.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = tf.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = tf.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)  # [node, dim_out]

        return new_h, new_c


class BiLSTM_(tf.keras.Model):
    def __init__(self, dim, return_seq=False):
        super(BiLSTM_, self).__init__()
        self.dim = dim
        # self.c_init_f = tfe.Variable(tf.get_variable("c_init_f", [1, dim], tf.float32,
        #                                              initializer=he_normal()))
        # self.h_init_f = tfe.Variable(tf.get_variable("h_initf", [1, dim], tf.float32,
        #                                              initializer=he_normal()))
        # self.c_init_b = tfe.Variable(tf.get_variable("c_init_b", [1, dim], tf.float32,
        #                                              initializer=he_normal()))
        # self.h_init_b = tfe.Variable(tf.get_variable("h_init_b", [1, dim], tf.float32,
        #                                              initializer=he_normal()))
        self.c_init_f = tfe.Variable(tf.random_normal([1, dim], stddev=0.01, dtype=tf.float32))
        self.h_init_f = tfe.Variable(tf.random_normal([1, dim], stddev=0.01, dtype=tf.float32))
        self.c_init_b = tfe.Variable(tf.random_normal([1, dim], stddev=0.01, dtype=tf.float32))
        self.h_init_b = tfe.Variable(tf.random_normal([1, dim], stddev=0.01, dtype=tf.float32))
        self.Cell_f = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(dim)
        self.Cell_b = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(dim)
        self.fc = tf.keras.layers.Dense(dim, use_bias=False)
        self.return_seq = return_seq

    def call(self, x, length):
        '''x: [batch, length, dim]'''
        batch = x.shape[0]
        ys, states = tf.nn.bidirectional_dynamic_rnn(self.Cell_f, self.Cell_b, x,
                                                     length,
                                                     tf.nn.rnn_cell.LSTMStateTuple(
                                                         tf.tile(self.c_init_f, [batch, 1]),
                                                         tf.tile(self.h_init_f, [batch, 1])),
                                                     tf.nn.rnn_cell.LSTMStateTuple(
                                                         tf.tile(self.c_init_b, [batch, 1]),
                                                         tf.tile(self.h_init_b, [batch, 1])))
        if self.return_seq:
            return self.fc(tf.concat(ys, -1))
        else:
            state_f, state_b = states
            state_concat = tf.concat([state_f.h, state_b.h], -1)
            return self.fc(state_concat)


class BiLSTM(tf.keras.Model):
    def __init__(self, dim, return_seq=False):
        super(BiLSTM, self).__init__()
        self.dim = dim
        self.c_init_f = tfe.Variable(tf.random_normal([1, dim], stddev=0.01, dtype=tf.float32))
        self.h_init_f = tfe.Variable(tf.random_normal([1, dim], stddev=0.01, dtype=tf.float32))
        self.c_init_b = tfe.Variable(tf.random_normal([1, dim], stddev=0.01, dtype=tf.float32))
        self.h_init_b = tfe.Variable(tf.random_normal([1, dim], stddev=0.01, dtype=tf.float32))
        self.lay_f = tf.keras.layers.CuDNNLSTM(dim, return_sequences=True, return_state=True)
        self.lay_b = tf.keras.layers.CuDNNLSTM(dim, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(dim, use_bias=False)
        self.return_seq = return_seq

    def call(self, x, length):
        '''x: [batch, length, dim]'''
        batch = x.shape[0]
        x_back = tf.reverse_sequence(x, length, 1)

        init_state_f = (tf.tile(self.h_init_f, [batch, 1]), tf.tile(self.c_init_f, [batch, 1]))
        init_state_b = (tf.tile(self.h_init_b, [batch, 1]), tf.tile(self.c_init_b, [batch, 1]))

        y_f, h_f, c_f = self.lay_f(x, init_state_f)
        y_b, h_b, c_b = self.lay_b(x_back, init_state_b)

        y = tf.concat([y_f, y_b], -1)

        if self.return_seq:
            return self.fc(y)
        else:
            y_last = tf.gather_nd(y, tf.stack([tf.range(batch), length - 1], 1))
            return self.fc(y_last)


class ShidoTreeLSTMLayer(tf.keras.Model):
    def __init__(self, dim_in, dim_out):
        super(ShidoTreeLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = BiLSTM(dim_out, return_seq=True)
        self.U_i = BiLSTM(dim_out)
        self.U_u = BiLSTM(dim_out)
        self.U_o = BiLSTM(dim_out)
        self.W = tf.keras.layers.Dense(dim_out * 4)
        # self.h_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        # self.c_init = tfe.Variable(
        #     tf.get_variable("c_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.h_init = tf.zeros([1, dim_out], tf.float32)
        self.c_init = tf.zeros([1, dim_out], tf.float32)

    def call(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice)
            res_h.append(h_tensor[:, :])
            res_c.append(c_tensor[:, :])
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
        return res_h, res_c

    def apply(self, x, h_tensor, c_tensor, indice):

        mask_bool = tf.not_equal(indice, -1.)
        mask = tf.cast(mask_bool, tf.float32)  # [nodes, child]
        length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        h = tf.gather(h_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))  # [nodes, child, dim]
        c = tf.gather(c_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = self.U_f(h, length)
        branch_f_k = tf.sigmoid(tf.expand_dims(W_f_x, 1) + branch_f_k)
        branch_f = tf.reduce_sum(branch_f_k * c * tf.expand_dims(mask, -1), 1)  # [node, dim_out]

        branch_i = self.U_i(h, length)  # [nodes, dim_out]
        branch_i = tf.sigmoid(branch_i + W_i_x)   # [nodes, dim_out]
        branch_u = self.U_u(h, length)  # [nodes, dim_out]
        branch_u = tf.tanh(branch_u + W_u_x)
        branch_o = self.U_o(h, length)  # [nodes, dim_out]
        branch_o = tf.sigmoid(branch_o + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)  # [node, dim_out]

        return new_h, new_c


class ShidoTreeLSTMLayerTreeBase(tf.keras.Model):
    def __init__(self, dim_in, dim_out):
        super(ShidoTreeLSTMLayerTreeBase, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = BiLSTM(dim_out, return_seq=True)
        self.U_i = BiLSTM(dim_out)
        self.U_u = BiLSTM(dim_out)
        self.U_o = BiLSTM(dim_out)
        self.W = tf.keras.layers.Dense(dim_out * 4)
        # self.h_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        # self.c_init = tfe.Variable(
        #     tf.get_variable("c_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.h_init = tf.zeros([1, dim_out], tf.float32)
        self.c_init = tf.zeros([1, dim_out], tf.float32)

    @staticmethod
    def get_nums(roots):
        res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
        max_len = max([len(x) for x in res])
        res = tf.keras.preprocessing.sequence.pad_sequences(
            res, max_len, padding="post", value=-1.)
        return tf.constant(res, tf.int32)

    def call(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch2(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        indices = [self.get_nums(nodes) for nodes in depthes]

        h_tensor = self.h_init
        c_tensor = self.c_init
        for indice, nodes in zip(indices, depthes):
            x = tf.stack([node.h for node in nodes])  # [nodes, dim_in]
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice, nodes)
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
        return depthes[-1]

    def apply(self, x, h_tensor, c_tensor, indice, nodes):

        mask_bool = tf.not_equal(indice, -1.)
        mask = tf.cast(mask_bool, tf.float32)  # [nodes, child]
        length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        h = tf.gather(h_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))  # [nodes, child, dim]
        c = tf.gather(c_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = self.U_f(h, length)
        branch_f_k = tf.sigmoid(tf.expand_dims(W_f_x, 1) + branch_f_k)
        branch_f = tf.reduce_sum(branch_f_k * c * tf.expand_dims(mask, -1), 1)  # [node, dim_out]

        branch_i = self.U_i(h, length)  # [nodes, dim_out]
        branch_i = tf.sigmoid(branch_i + W_i_x)   # [nodes, dim_out]
        branch_u = self.U_u(h, length)  # [nodes, dim_out]
        branch_u = tf.tanh(branch_u + W_u_x)
        branch_o = self.U_o(h, length)  # [nodes, dim_out]
        branch_o = tf.sigmoid(branch_o + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)  # [node, dim_out]

        for n, c, h in zip(nodes, new_c, new_h):
            n.c = c
            n.h = h

        return new_h, new_c


class ShidoTreeLSTMWithEmbedding(ShidoTreeLSTMLayer):
    def __init__(self, in_vocab, dim_in, dim_out):
        super(ShidoTreeLSTMWithEmbedding, self).__init__(dim_in, dim_out)
        self.E = tf.get_variable("E", [in_vocab, dim_in], tf.float32,
                                 initializer=tf.keras.initializers.RandomUniform())
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = BiLSTM(dim_out, return_seq=True)
        self.U_i = BiLSTM(dim_out)
        self.U_u = BiLSTM(dim_out)
        self.U_o = BiLSTM(dim_out)
        self.W = tf.keras.layers.Dense(dim_out * 4)
        # self.h_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        # self.c_init = tfe.Variable(
        #     tf.get_variable("c_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.h_init = tf.zeros([1, dim_out], tf.float32)
        self.c_init = tf.zeros([1, dim_out], tf.float32)

    def call(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch2(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        indices = [self.get_nums(nodes) for nodes in depthes]

        h_tensor = self.h_init
        c_tensor = self.c_init
        for indice, nodes in zip(indices, depthes):
            x = tf.nn.embedding_lookup(self.E, [node.label for node in nodes])  # [nodes, dim_in]
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice, nodes)
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
        return depthes[-1]


class TreeDropout(tf.keras.Model):
    def __init__(self, rate):
        super(TreeDropout, self).__init__()
        self.dropout_layer = tf.keras.layers.Dropout(rate)

    def call(self, roots):
        nodes = [node for root in roots for node in traverse(root)]
        ys = [node.h for node in nodes]
        tensor = tf.stack(ys)
        dropped = self.dropout_layer(tensor)
        for e, v in enumerate(tf.split(dropped, len(ys))):
            nodes[e].h = tf.squeeze(v)
        return roots


class SetEmbeddingLayer(tf.keras.Model):
    def __init__(self, dim_E, in_vocab):
        super(SetEmbeddingLayer, self).__init__()
        self.E = tf.get_variable("E", [in_vocab, dim_E], tf.float32,
                                 initializer=tf.keras.initializers.RandomUniform())

    def call(self, sets):
        length = [len(s) for s in sets]
        concatenated = tf.concat(sets, 0)
        embedded = tf.nn.embedding_lookup(self.E, concatenated)
        y = tf.split(embedded, length)
        return y


class LSTMEncoder(tf.keras.Model):
    def __init__(self, dim, layer=1):
        super(LSTMEncoder, self).__init__()
        self.dim = dim
        # self.c_init_f = tfe.Variable(tf.get_variable("c_init_f", [1, dim], tf.float32,
        #                                              initializer=he_normal()))
        # self.h_init_f = tfe.Variable(tf.get_variable("h_initf", [1, dim], tf.float32,
        #                                              initializer=he_normal()))
        self.Cell_f = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(dim)
        self.h_init_f = tf.zeros([1, dim], tf.float32)
        self.c_init_f = tf.zeros([1, dim], tf.float32)

    def call(self, x, length):
        '''x: [batch, length, dim]'''
        batch = x.shape[0]
        ys, states = tf.nn.dynamic_rnn(self.Cell_f, x,
                                       length,
                                       tf.nn.rnn_cell.LSTMStateTuple(
                                           tf.tile(self.c_init_f, [batch, 1]),
                                           tf.tile(self.h_init_f, [batch, 1])))
        return ys, states


class SequenceEmbeddingLayer(tf.keras.Model):
    def __init__(self, dim_E, in_vocab):
        super(SequenceEmbeddingLayer, self).__init__()
        self.E = tf.keras.layers.Embedding(in_vocab, dim_E)

    def call(self, y):
        y = self.E(y)
        return y
