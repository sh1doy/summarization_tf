import tensorflow as tf
from utils import pad_tensor
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
        dec_hidden = h
        dec_cell = c
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
