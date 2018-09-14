"""layers"""

import chainer
from utils import *
from chainer import links as L
from chainer import functions as F
from chainer import Chain
from chainer import initializers
import numpy
xp = numpy


class ChildSumLSTMLayer_old(Chain):
    def __init__(self, dim_in, dim_out, dropout=0.0):
        super(ChildSumLSTMLayer_old, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        with self.init_scope():
            self.U_f = L.Linear(dim_out, dim_out, nobias=True)
            self.U_i = L.Linear(dim_out, dim_out, nobias=True)
            self.U_u = L.Linear(dim_out, dim_out, nobias=True)
            self.U_o = L.Linear(dim_out, dim_out, nobias=True)
            self.W = L.Linear(dim_in, dim_out * 4)
            self.h_init = chainer.Parameter(chainer.initializers.HeNormal(), (1, dim_out))

    def __call__(self, root):
        return self.apply(root, None)

    def apply(self, node, parent):
        x = node.h
        children = [self.apply(child, node) for child in node.children]
        h_sum = F.add(*[child.h for child in children]
                      ) if children != [] else self.h_init[0]
        W_x = self.W(F.expand_dims(x, 0))[0]
        W_f_x = W_x[:self.dim_out]
        W_i_x = W_x[self.dim_out:self.dim_out * 2]
        W_u_x = W_x[self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[self.dim_out * 3:]
        branch_f_k = [F.sigmoid(self.U_f(F.expand_dims(child.h, 0))[0] + W_f_x)
                      for child in children]
        branch_i = F.sigmoid(self.U_i(F.expand_dims(h_sum, 0))[0] + W_i_x)
        branch_u = F.tanh(self.U_u(F.expand_dims(h_sum, 0))[0] + W_u_x)
        branch_o = F.sigmoid(self.U_o(F.expand_dims(h_sum, 0))[0] + W_o_x)
        if children != []:
            new_c = branch_i * branch_u + \
                F.add(*[child.c * f_k for child, f_k in zip(children, branch_f_k)])
        else:
            new_c = branch_i * branch_u
        new_h = branch_o * F.tanh(new_c)

        # Ascending
        return TreeLSTMNode(new_h, new_c, parent=parent, children=children, num=node.num)


class ChildSumLSTMLayer(Chain):
    def __init__(self, dim_in, dim_out, dropout=0.0):
        super(ChildSumLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        with self.init_scope():
            self.U_f = L.Linear(dim_out, dim_out, nobias=True)
            self.U_iuo = L.Linear(dim_out, dim_out * 3, nobias=True)
            self.W = L.Linear(dim_in, dim_out * 4)
            self.h_init = chainer.Parameter(chainer.initializers.HeNormal(), (1, dim_out))

    def get_child_c_tensor(self, node):
        '''return: [children, dim_out]'''
        if node.children == []:
            return self.xp.zeros((1, self.dim_out), "float32")
        else:
            return F.stack([child.c for child in node.children])

    def get_child_h_tensor(self, node):
        '''return: [children, dim_out]'''
        if node.children == []:
            return self.h_init
        else:
            return F.stack([child.h for child in node.children])

    def __call__(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        for nodes in depthes:
            self.apply(nodes)
        return depthes[-1]

    def apply(self, nodes):
        child_hs = [self.get_child_h_tensor(node) for node in nodes]  # (node, [child, dim_out])
        child_cs = [self.get_child_c_tensor(node) for node in nodes]  # (node, [child, dim_out])

        h_sum = F.stack([F.sum(h, 0) for h in child_hs])  # [nodes, dim_out]
        x = F.stack([node.h for node in nodes])  # [nodes, dim_in]
        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = sequence_apply(self.U_f, child_hs)  # (node, [child, dim_out])
        branch_f_k = [F.sigmoid(F.broadcast_to(W_f_x[e:e + 1], h.shape) + h)
                      for e, h in enumerate(branch_f_k)]
        branch_f = F.stack([F.sum(f_k * c, 0)
                            for f_k, c in zip(branch_f_k, child_cs)])  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = F.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = F.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = F.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * F.tanh(new_c)  # [node, dim_out]

        for n, c, h in zip(nodes, new_c, new_h):
            n.c = c
            n.h = h


class BiLSTM(Chain):

    def __init__(self, dim, return_single=True, dropout=0.0):
        super(BiLSTM, self).__init__()
        with self.init_scope():
            self.c_init_forward = chainer.Parameter(chainer.initializers.HeNormal(), (1, dim))
            self.h_init_forward = chainer.Parameter(chainer.initializers.HeNormal(), (1, dim))
            self.c_init_backward = chainer.Parameter(chainer.initializers.HeNormal(), (1, dim))
            self.h_init_backward = chainer.Parameter(chainer.initializers.HeNormal(), (1, dim))
            self.linear = L.Linear(2 * dim, dim, nobias=True)
            self.LSTM_forward = L.StatelessLSTM(dim, dim,
                                                chainer.initializers.GlorotNormal(),
                                                chainer.initializers.GlorotNormal())
            self.LSTM_backward = L.StatelessLSTM(dim, dim,
                                                 chainer.initializers.GlorotNormal(),
                                                 chainer.initializers.GlorotNormal())
        self.return_single = return_single

    def __call__(self, seq):
        seq = [F.expand_dims(vec, 0) for vec in seq]

        c_forward = self.c_init_forward
        h_forward = self.h_init_forward
        hs_forward = []
        for vec in seq:
            c_forward, h_forward = self.LSTM_forward(c_forward, h_forward, vec)
            hs_forward.append(h_forward)

        c_backward = self.c_init_backward
        h_backward = self.h_init_backward
        hs_backward = []
        for vec in seq[::-1]:
            c_backward, h_backward = self.LSTM_backward(c_backward, h_backward, vec)
            hs_backward.append(h_backward)

        if self.return_single:
            h_concat = F.concat([hs_forward[-1], hs_backward[-1]])
            y = self.linear(h_concat)[0]
        else:
            hs_forward = F.concat(hs_forward, 0)
            hs_backward = F.concat(hs_backward[::-1], 0)
            h_concat = F.concat([hs_forward, hs_backward], -1)
            y = self.linear(h_concat)

        return y


class BiLSTM_cudnn(Chain):

    def __init__(self, dim, return_single=True, dropout=0.0):
        super(BiLSTM_cudnn, self).__init__()
        with self.init_scope():
            self.c_init = chainer.Parameter(chainer.initializers.HeNormal(), (2, 1, dim))
            self.h_init = chainer.Parameter(chainer.initializers.HeNormal(), (2, 1, dim))
            self.linear = L.Linear(2 * dim, dim, nobias=True)
            self.LSTM = L.NStepBiLSTM(1, dim, dim, dropout)
        self.return_single = return_single
        self.dim = dim

    def __call__(self, seq):
        seq = [F.stack(seq)]
        hy, cy, ys = self.LSTM(self.h_init, self.c_init, seq)

        if self.return_single:
            h_concat = F.concat([ys[0][-1:][:self.dim], ys[0][:1][self.dim:]], 0)
            y = self.linear(h_concat)[0]
        else:
            y = self.linear(ys[0])

        return y


class ShidoTreeLSTM(Chain):
    def __init__(self, dim_in, dim_out, dropout=0.0, trainable_init=True):
        super(ShidoTreeLSTM, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        with self.init_scope():
            self.U_f = BiLSTM(dim_out, return_single=False)
            self.U_i = BiLSTM(dim_out, return_single=True)
            self.U_u = BiLSTM(dim_out, return_single=True)
            self.U_o = BiLSTM(dim_out, return_single=True)
            self.W = L.Linear(dim_in, dim_out * 4)
            self.h_init = chainer.Parameter(chainer.initializers.HeNormal(), (1, dim_out))

    def __call__(self, root):
        return self.apply(root, None)

    def apply(self, node, parent):
        x = node.h
        children = [self.apply(child, node) for child in node.children]
        children_h = [child.h for child in children] if children != [] else [self.h_init[0]]
        W_x = self.W(F.expand_dims(x, 0))[0]
        W_f_x = W_x[:self.dim_out]
        W_i_x = W_x[self.dim_out:self.dim_out * 2]
        W_u_x = W_x[self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[self.dim_out * 3:]
        branch_f_k = [F.sigmoid(c + W_f_x) for c in self.U_f(children_h)]
        branch_i = F.sigmoid(self.U_i(children_h) + W_i_x)
        branch_u = F.tanh(self.U_u(children_h) + W_u_x)
        branch_o = F.sigmoid(self.U_o(children_h) + W_o_x)
        if children != []:
            new_c = branch_i * branch_u + \
                F.add(*[child.c * f_k for child, f_k in zip(children, branch_f_k)])
        else:
            new_c = branch_i * branch_u
        new_h = branch_o * F.tanh(new_c)

        # Ascending
        return TreeLSTMNode(new_h, new_c, parent=parent, children=children, num=node.num)


class TreeEmbeddingLayer(Chain):
    def __init__(self, dim_E, in_vocab):
        super(TreeEmbeddingLayer, self).__init__()
        with self.init_scope():
            self.E = L.EmbedID(in_vocab + 1, dim_E, initializers.HeNormal(), -1)

    def __call__(self, root):
        labels = traverse_label(root)
        embedded = self.E(self.xp.array(labels))
        new_nodes = self.Node2TreeLSTMNode(root, parent=None)
        for rep, node in zip(embedded, traverse(new_nodes)):
            node.h = rep
        return new_nodes

    def Node2TreeLSTMNode(self, node, parent):
        children = [self.Node2TreeLSTMNode(c, node) for c in node.children]
        return TreeLSTMNode(node.label, parent=parent, children=children, num=node.num)


def sequence_embed(embed, xs):
    x_len = [len(x) for x in xs]
    x_section = numpy.cumsum(x_len[:-1])
    ex = embed(F.concat(xs, axis=0))
    exs = F.split_axis(ex, x_section, 0)
    return exs


class BiLSTM_Encoder(Chain):

    def __init__(self, n_layers, dim_E, dim_rep, dropout=0.5):
        super(BiLSTM_Encoder, self).__init__()
        with self.init_scope():
            self.LSTM = L.NStepBiLSTM(n_layers, dim_E, dim_rep, dropout)
            self.hh = [L.Linear(dim_rep * 2, dim_rep) for _ in range(n_layers)]
            self.cc = [L.Linear(dim_rep * 2, dim_rep) for _ in range(n_layers)]
            for e, link in enumerate(self.hh):
                self.add_link(str(e) + "h", link)
            for e, link in enumerate(self.cc):
                self.add_link(str(e) + "c", link)
            self.yy = L.Linear(dim_rep * 2, dim_rep)
            self.n_layers = n_layers

    def __call__(self, hx, cx, xs):
        hy, cy, ys = self.LSTM(hx, cx, xs)
        hy = F.stack([self.hh[i](
            F.concat([hy[i * 2], hy[i * 2 + 1]], -1)) for i in range(self.n_layers)], 0)
        cy = F.stack([self.cc[i](
            F.concat([cy[i * 2], cy[i * 2 + 1]], -1)) for i in range(self.n_layers)], 0)
        ys_len = [len(y) for y in ys]
        y_section = numpy.cumsum(ys_len[:-1])
        yys = self.yy(F.concat(ys, 0))
        ys = F.split_axis(yys, y_section, 0)
        return hy, cy, ys


class Attention(Chain):
    '''Implement of global attention (2014, Bahdanau+)'''

    def __init__(self, hidden_size, dropout=0.0):
        super(Attention, self).__init__()
        with self.init_scope():
            self.eh = L.Linear(hidden_size, hidden_size)  # for inputs from encoder
            self.hh = L.Linear(hidden_size, hidden_size)  # for inputs from decoder
            self.hw = L.Linear(hidden_size, 1)
            self.hidden_size = hidden_size
            self.dropout = dropout

    def __call__(self, y_enc, h_dec):
        '''
        y_enc: list(array(len, dim))
        h_dec: array(batch, dim)
        '''
        batch_size = h_dec.data.shape[0]
        y_len = [len(y) for y in y_enc]
        y_section = numpy.cumsum(y_len[:-1])
        y_eh = self.eh(F.concat(y_enc, 0))  # [batch*len, hid]
        y_eh = F.split_axis(y_eh, y_section, 0)
        mask = get_sequence_mask(y_eh, self.xp)
        y_eh = F.pad_sequence(y_eh)  # [batch, len, hid]
        h_hh = self.hh(h_dec)  # [batch, hid]
        h_hh = F.broadcast_to(F.expand_dims(h_hh, 1), y_eh.shape)  # [batch, len, hid]
        keys = F.tanh(y_eh + h_hh)
        keys = F.dropout(keys, self.dropout)
        keys = F.reshape(keys, shape=[-1, self.hidden_size])  # [batch*len, hid]
        keys = self.hw(keys)  # [batch*len, 1]
        keys = F.reshape(keys, shape=[batch_size, -1])  # [batch, len]
        keys = F.where(
            mask,
            keys,
            self.xp.ones(keys.shape, "float32") * self.xp.float32("-inf"))  # mask for softmax
        keys = F.softmax(keys)
        keys = F.expand_dims(keys, -1)
        attention = F.broadcast_to(keys, y_eh.shape) * F.pad_sequence(y_enc, padding=0)
        attention = F.sum(attention, axis=1)  # [batch, hid]
        return attention


class LSTM_Attention_Decoder(Chain):
    def __init__(self, n_layers, dim_F, dim_rep, dropout=0.5):
        super(LSTM_Attention_Decoder, self).__init__()
        with self.init_scope():
            self.LSTM = L.NStepLSTM(n_layers, dim_F, dim_rep, dropout)
            self.n_layers = n_layers
            self.attention = Attention(dim_rep, dropout)

    def __call__(self, hx, cx, y_enc, y_in):
        hy, cy, ys = self.LSTM(hx, cx, y_in)
        length = [len(x) for x in ys]
        ys = F.pad_sequence(ys)  # [batch, len, F_dim]
        result = []
        for i in range(ys.shape[1]):
            result.append(self.attention(y_enc, ys[:, i]))
        result = F.stack(result, 0)  # [len, batch, hid]
        result = F.swapaxes(result, 0, 1)  # [batch, len, hid]
        result = F.separate(result)
        result = [x[:i] for x, i in zip(result, length)]  # same as y_in

        return hy, cy, result
