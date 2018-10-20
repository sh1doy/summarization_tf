import tensorflow as tf
from utils import pad_tensor
from layers import *
import numpy as np


class AttentionDecoder(tf.keras.Model):
    def __init__(self, dim_F, dim_rep, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.dim_rep = dim_rep
        self.F = tf.keras.layers.Embedding(vocab_size, dim_F)
        self.gru = tf.keras.layers.CuDNNLSTM(dim_rep,
                                             return_sequences=True,
                                             return_state=True,
                                             recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dim_rep)
        self.W2 = tf.keras.layers.Dense(self.dim_rep)
        self.V = tf.keras.layers.Dense(1)

    @staticmethod
    def loss_function(real, pred):
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred)
        return tf.reduce_sum(loss_)

    def get_loss(self, enc_y, states, target, dropout=0.0):
        '''
        enc_y: batch_size([seq_len, dim])
        states: ([batch, dim], [batch, dim])
        target: [batch, max_len] (padded with -1.)
        '''
        mask = tf.not_equal(target, -1.)
        h, c = states
        enc_y, _ = pad_tensor(enc_y)
        enc_y = tf.nn.dropout(enc_y, 1. - dropout)
        dec_hidden = tf.nn.dropout(h, 1. - dropout)
        dec_cell = tf.nn.dropout(c, 1. - dropout)
        dec_input = target[:, 0]
        loss = 0
        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, dec_cell, att = self.call(
                dec_input, dec_hidden, dec_cell, enc_y)
            real = tf.boolean_mask(target[:, t], mask[:, t])
            pred = tf.boolean_mask(predictions, mask[:, t])
            loss += self.loss_function(real, pred)
            # using teacher forcing
            dec_input = target[:, t]

        return loss / tf.reduce_sum(tf.cast(mask, tf.float32))

    def translate(self, y_enc, states, max_length, start_token, end_token):
        '''
        enc_y: [seq_len, dim]
        states: ([dim,], [dim,])
        '''
        attention_plot = np.zeros((max_length, y_enc.shape[0]))

        h, c = states
        y_enc = tf.expand_dims(y_enc, 0)
        dec_hidden = tf.expand_dims(h, 0)
        dec_cell = tf.expand_dims(c, 0)
        dec_input = tf.constant(start_token, tf.int32, [1])
        result = []

        for t in range(max_length):
            predictions, dec_hidden, dec_cell, attention_weights = self.call(
                dec_input, dec_hidden, dec_cell, y_enc)

            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(predicted_id)

            if predicted_id == end_token:
                return result[:-1], attention_plot[:t]

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims(predicted_id, 0)

        return result, attention_plot

    def call(self, x, hidden, cell, enc_y):
        # enc_y shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = tf.nn.tanh(self.W1(enc_y) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_y
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = tf.expand_dims(self.F(x), 1)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state, cell = self.gru(x, (hidden, cell))

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, cell, attention_weights


class BaseModel(tf.keras.Model):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0., lr=1e-3):
        super(BaseModel, self).__init__()
        self.dim_E = dim_E
        self.dim_F = dim_F
        self.dim_rep = dim_rep
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.dropout = dropout
        self.decoder = AttentionDecoder(dim_F, dim_rep, out_vocab)
        self.optimizer = tf.train.AdamOptimizer(lr)

    def encode(self, trees):
        '''
        ys: list of [seq_len, dim]
        hx, cx: [batch, dim]
        return: ys, [hx, cx]
        '''

    def train_on_batch(self, x, y):
        with tf.GradientTape() as tape:
            y_enc, (c, h) = self.encode(x)
            loss = self.decoder.get_loss(y_enc, (c, h), y, dropout=self.dropout)
            variables = self.variables
            gradients = tape.gradient(loss, variables)
            self.optimizer.apply_gradients(zip(gradients, variables))
        return loss.numpy()

    def translate(self, x, nl_i2w, nl_w2i, max_length=100):
        res = []
        y_enc, (c, h) = self.encode(x)
        for i in range(len(x) if type(x) is list else x.shape[0]):
            nl, _ = self.decoder.translate(
                y_enc[i], (c[i], h[i]), max_length, nl_w2i["<s>"], nl_w2i["</s>"])
            res.append([nl_i2w[n] for n in nl])
        return res

    def evaluate_on_batch(self, x, y):
        y_enc, (c, h) = self.encode(x)
        loss = self.decoder.get_loss(y_enc, (c, h), y)
        return loss.numpy()


class CodennModel(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-3):
        super(CodennModel, self).__init__(dim_E, dim_F, dim_rep, in_vocab,
                                          out_vocab, layer, dropout, lr)
        self.E = SetEmbeddingLayer(dim_E, in_vocab)

    def encode(self, sets):
        sets = self.E(sets)

        hx = tf.zeros([len(sets), self.dim_rep])
        cx = tf.zeros([len(sets), self.dim_rep])
        ys = sets

        return ys, [hx, cx]


class Seq2seqModel(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-3):
        super(Seq2seqModel, self).__init__(dim_E, dim_F,
                                           dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.E = SequenceEmbeddingLayer(dim_E, in_vocab)
        self.encoder = LSTMEncoder(dim_E, dim_rep)

    def encode(self, seq):
        length = get_length(seq)
        seq = self.E(seq)
        ys, states = self.encoder(seq, length)

        cx = states.c
        hx = states.h
        ys = [y[:i] for y, i in zip(tf.unstack(ys, axis=0), length.numpy())]

        return ys, [hx, cx]


class ChildsumModel_old(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-4):
        super(ChildsumModel_old, self).__init__(dim_E, dim_F,
                                                dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        self.encoder = ChildSumLSTMLayer(dim_E, dim_rep)

    def encode(self, trees):
        trees = self.E(trees)
        trees = self.encoder(trees)

        hx = tf.stack([tree.h for tree in trees])
        cx = tf.stack([tree.c for tree in trees])
        ys = [tf.stack([node.h for node in traverse(tree)]) for tree in trees]

        return ys, [hx, cx]


class ChildsumModel(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-4):
        super(ChildsumModel, self).__init__(dim_E, dim_F,
                                            dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.encoder = ChildSumLSTMLayerWithEmbedding(in_vocab, dim_E, dim_rep)

    def encode(self, trees):
        trees = self.encoder(trees)

        hx = tf.stack([tree.h for tree in trees])
        cx = tf.stack([tree.c for tree in trees])
        ys = [tf.stack([node.h for node in traverse(tree)]) for tree in trees]

        return ys, [hx, cx]


class MultiwayModel_old(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.0, lr=1e-4):
        super(MultiwayModel_old, self).__init__(dim_E, dim_F,
                                                dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        self.encoder = ShidoTreeLSTM(dim_E, dim_rep)

    def encode(self, trees):
        trees = self.E(trees)
        trees = self.encoder(trees)

        hx = tf.stack([tree.h for tree in trees])
        cx = tf.stack([tree.c for tree in trees])
        ys = [tf.stack([node.h for node in traverse(tree)]) for tree in trees]

        return ys, [hx, cx]


class MultiwayModel(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.0, lr=1e-4):
        super(MultiwayModel, self).__init__(dim_E, dim_F,
                                            dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.encoder = ShidoTreeLSTMWithEmbedding(in_vocab, dim_E, dim_rep)

    def encode(self, trees):
        trees = self.encoder(trees)

        hx = tf.stack([tree.h for tree in trees])
        cx = tf.stack([tree.c for tree in trees])
        ys = [tf.stack([node.h for node in traverse(tree)]) for tree in trees]

        return ys, [hx, cx]
