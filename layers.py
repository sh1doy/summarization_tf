"""layers"""

import tensorflow as tf
from utils import *
tfe = tf.contrib.eager


class TreeEmbeddingLayer(tf.keras.Model):
    def __init__(self, dim_E, in_vocab):
        super(TreeEmbeddingLayer, self).__init__()
        self.E = tf.get_variable("E", [in_vocab, dim_E], tf.float32,
                                 initializer=he_normal())

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


class ChildSumLSTMLayer2(tf.keras.Model):
    def __init__(self, dim_in, dim_out):
        super(ChildSumLSTMLayer2, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = tf.keras.layers.Dense(dim_out, use_bias=False)
        self.U_iuo = tf.keras.layers.Dense(dim_out * 3, use_bias=False)
        self.W = tf.keras.layers.Dense(dim_out * 4, use_bias=False)
        self.h_init = tfe.Variable(
            tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.c_init = tf.constant(0., tf.float32, [1, self.dim_out])

    def get_child_c_tensor(self, node):
        '''return: [children, dim_out]'''
        if node.children == []:
            return self.c_init
        else:
            return tf.stack([child.c for child in node.children])

    def get_child_h_tensor(self, node):
        '''return: [children, dim_out]'''
        if node.children == []:
            return self.h_init
        else:
            return tf.stack([child.h for child in node.children])

    def call(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        for nodes in depthes:
            self.apply(nodes)
        return depthes[-1]

    def apply(self, nodes):
        child_hs = [self.get_child_h_tensor(node) for node in nodes]  # (node, [child, dim_out])
        child_cs = [self.get_child_c_tensor(node) for node in nodes]  # (node, [child, dim_out])

        h_sum = tf.stack([tf.reduce_sum(h, 0) for h in child_hs])  # [nodes, dim_out]
        x = tf.stack([node.h for node in nodes])  # [nodes, dim_in]
        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = sequence_apply(self.U_f, child_hs)  # (node, [child, dim_out])
        branch_f_k = [tf.sigmoid(W_f_x[e] + h) for e, h in enumerate(branch_f_k)]
        branch_f = tf.stack([tf.reduce_sum(f_k * c, 0)
                             for f_k, c in zip(branch_f_k, child_cs)])  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = tf.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = tf.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = tf.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)  # [node, dim_out]

        for n, c, h in zip(nodes, new_c, new_h):
            n.c = c
            n.h = h


class ChildSumLSTMLayer(tf.keras.Model):
    def __init__(self, dim_in, dim_out):
        super(ChildSumLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = tf.keras.layers.Dense(dim_out, use_bias=False)
        self.U_iuo = tf.keras.layers.Dense(dim_out * 3, use_bias=False)
        self.W = tf.keras.layers.Dense(dim_out * 4, use_bias=False)
        self.h_init = tfe.Variable(
            tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.c_init = tf.constant(0., tf.float32, [1, self.dim_out])

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
