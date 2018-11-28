import tensorflow as tf
from utils import pad_tensor
from layers import *
import numpy as np


class AttentionDecoder(tf.keras.Model):
    def __init__(self, dim_F, dim_rep, vocab_size, layer=1):
        super(AttentionDecoder, self).__init__()
        self.layer = layer
        self.dim_rep = dim_rep
        self.F = tf.keras.layers.Embedding(vocab_size, dim_F)
        for i in range(layer):
            self.__setattr__("layer{}".format(i),
                             tf.keras.layers.CuDNNLSTM(dim_rep,
                                                       return_sequences=True,
                                                       return_state=True,
                                                       recurrent_initializer='glorot_uniform'))
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dim_rep)
        self.W2 = tf.keras.layers.Dense(self.dim_rep)
        self.V = tf.keras.layers.Dense(1)
        print("I am Decoder, dim is {} and {} layered".format(str(self.dim_rep), str(self.layer)))

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

        l_states = [(dec_hidden, dec_cell) for _ in range(self.layer)]
        target = tf.nn.relu(target)
        dec_input = target[:, 0]
        loss = 0
        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predictions, l_states, att = self.call(
                dec_input, l_states, enc_y)
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

        l_states = [(dec_hidden, dec_cell) for _ in range(self.layer)]

        for t in range(max_length):
            predictions, l_states, attention_weights = self.call(
                dec_input, l_states, y_enc)

            attention_weights = tf.reshape(attention_weights, (-1,))
            attention_plot[t] = attention_weights.numpy()

            predicted_id = tf.argmax(predictions[0]).numpy()
            result.append(predicted_id)

            if predicted_id == end_token:
                return result[:-1], attention_plot[:t]

            # the predicted ID is fed back into the model
            dec_input = tf.expand_dims(predicted_id, 0)

        return result, attention_plot

    def call(self, x, l_states, enc_y):
        # enc_y shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(l_states[-1][0], 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = tf.nn.tanh(self.W1(enc_y) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_y
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = tf.expand_dims(x, 1)
        x = self.F(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        new_l_states = []
        for i, states in zip(range(self.layer), l_states):
            if i < self.layer - 1:
                skip = x
                x, h, c = getattr(self, "layer{}".format(i))(x, states)
                x += skip
            else:
                x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
                x, h, c = getattr(self, "layer{}".format(i))(x, states)
            n_states = (h, c)
            new_l_states.append(n_states)

        # output shape == (batch_size * 1, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(x)

        return x, new_l_states, attention_weights


class BaseModel(tf.keras.Model):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0., lr=1e-3):
        super(BaseModel, self).__init__()
        self.dim_E = dim_E
        self.dim_F = dim_F
        self.dim_rep = dim_rep
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.dropout = dropout
        self.decoder = AttentionDecoder(dim_F, dim_rep, out_vocab, layer)
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
        batch_size = len(y_enc)
        for i in range(batch_size):
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
        self.dropout = dropout
        self.E = SetEmbeddingLayer(dim_E, in_vocab)
        print("I am CodeNNModel, dim is {} and {} layered".format(
            str(self.dim_rep), "0"))

    def encode(self, sets):
        sets = self.E(sets)
        # sets = [tf.nn.dropout(t, 1. - self.dropout) for t in sets]

        hx = tf.zeros([len(sets), self.dim_rep])
        cx = tf.zeros([len(sets), self.dim_rep])
        ys = sets

        return ys, [hx, cx]


class Seq2seqModel(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-3):
        super(Seq2seqModel, self).__init__(dim_E, dim_F,
                                           dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.layer = layer
        self.dropout = dropout
        self.E = tf.keras.layers.Embedding(in_vocab + 1, dim_E, mask_zero=True)
        for i in range(layer):
            self.__setattr__("layer{}".format(i),
                             tf.keras.layers.CuDNNLSTM(dim_rep,
                                                       return_sequences=True,
                                                       return_state=True))
        print("I am seq2seq model, dim is {} and {} layered".format(
            str(self.dim_rep), str(self.layer)))

    def encode(self, seq):
        length = get_length(seq)
        tensor = self.E(seq + 1)
        # tensor = tf.nn.dropout(tensor, 1. - self.dropout)
        for i in range(self.layer):
            skip = tensor
            tensor, h, c = getattr(self, "layer{}".format(i))(tensor)
            tensor += skip

        cx = c
        hx = h
        ys = [y[:i] for y, i in zip(tf.unstack(tensor, axis=0), length.numpy())]

        return ys, [hx, cx]


class ChildsumModel(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-4):
        super(ChildsumModel, self).__init__(dim_E, dim_F,
                                            dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.layer = layer
        self.dropout = dropout
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        for i in range(layer):
            self.__setattr__("layer{}".format(i), ChildSumLSTMLayer(dim_E, dim_rep))
        print("I am Child-sum model, dim is {} and {} layered".format(
            str(self.dim_rep), str(self.layer)))

    def encode(self, x):
        tensor, indice, tree_num = x
        tensor = self.E(tensor)
        # tensor = [tf.nn.dropout(t, 1. - self.dropout) for t in tensor]
        for i in range(self.layer):
            skip = tensor
            tensor, c = getattr(self, "layer{}".format(i))(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]
        cx = c[-1]
        ys = []
        batch_size = tensor[-1].shape[0]
        tensor = tf.concat(tensor, 0)
        tree_num = tf.concat(tree_num, 0)
        for batch in range(batch_size):
            ys.append(tf.boolean_mask(tensor, tf.equal(tree_num, batch)))
        return ys, [hx, cx]


class NaryModel(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-4):
        super(NaryModel, self).__init__(dim_E, dim_F,
                                        dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.layer = layer
        self.dropout = dropout
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        for i in range(layer):
            self.__setattr__("layer{}".format(i), NaryLSTMLayer(dim_E, dim_rep))
        print("I am N-ary model, dim is {} and {} layered".format(
            str(self.dim_rep), str(self.layer)))

    def encode(self, x):
        tensor, indice, tree_num = x
        tensor = self.E(tensor)
        # tensor = [tf.nn.dropout(t, 1. - self.dropout) for t in tensor]
        for i in range(self.layer):
            skip = tensor
            tensor, c = getattr(self, "layer{}".format(i))(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]
        cx = c[-1]
        ys = []
        batch_size = tensor[-1].shape[0]
        tensor = tf.concat(tensor, 0)
        tree_num = tf.concat(tree_num, 0)
        for batch in range(batch_size):
            ys.append(tf.boolean_mask(tensor, tf.equal(tree_num, batch)))
        return ys, [hx, cx]


class MultiwayModel(BaseModel):
    def __init__(self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.0, lr=1e-4):
        super(MultiwayModel, self).__init__(dim_E, dim_F,
                                            dim_rep, in_vocab, out_vocab, layer, dropout, lr)
        self.layer = layer
        self.dropout = dropout
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        for i in range(layer):
            self.__setattr__("layer{}".format(i), ShidoTreeLSTMLayer(dim_E, dim_rep))
        print("I am Multi-way model, dim is {} and {} layered".format(
            str(self.dim_rep), str(self.layer)))

    def encode(self, x):
        tensor, indice, tree_num = x
        tensor = self.E(tensor)
        # tensor = [tf.nn.dropout(t, 1. - self.dropout) for t in tensor]
        for i in range(self.layer):
            skip = tensor
            tensor, c = getattr(self, "layer{}".format(i))(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]
        cx = c[-1]
        ys = []
        batch_size = tensor[-1].shape[0]
        tensor = tf.concat(tensor, 0)
        tree_num = tf.concat(tree_num, 0)
        for batch in range(batch_size):
            ys.append(tf.boolean_mask(tensor, tf.equal(tree_num, batch)))
        return ys, [hx, cx]
