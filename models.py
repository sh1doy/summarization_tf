from layers import *
import chainer
from chainer import functions as F
from chainer import Chain
from chainer import cuda


class BaseModel(Chain):
    '''
    Write and self.decoder and self.encode() method.
    self.encode(): inputs -> hx, cx, ys
    self.decoder: hx, cx, ys -> hx, cx, ys
    '''

    def __init__(self, in_vocab, out_vocab):
        pass

    def encode(self, seq):
        """
        labels, relations, coefs: array
        texts: list of list
        """

        # Encode
        seq = [s[::-1] for s in seq]
        seq_embedded = sequence_embed(self.E, seq)
        hx, cx, ys = self.encoder(None, None, seq_embedded)
        return(hx, cx, ys)

    def get_loss(self, seq, texts):

        hx, cx, ys = self.encode(seq)

        # Decode
        eos = self.xp.array([0], 'i')
        ys_in = [F.concat([eos, text], axis=0) for text in texts]
        ys_out = [F.concat([text, eos], axis=0) for text in texts]

        texts_embed = sequence_embed(self.F, ys_in)
        batch = len(texts)

        _, _, os = self.decoder(hx, cx, ys, texts_embed)  # cx: (n_layers, batch, dim) <- WTF

        # Loss
        concat_os = F.concat(os, axis=0)
        concat_ys_out = F.concat(ys_out, axis=0)
        loss = F.sum(F.softmax_cross_entropy(
            self.W(concat_os), concat_ys_out, reduce='no')) / batch

        return(loss)

    def translate(self, seq, max_length=100):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            batch = len(seq)
            hx, cx, y_enc = self.encode(seq)
            ys = self.xp.full(batch, 0, 'i')
            h, c = hx, cx
            result = []
            for i in range(max_length):
                eys = self.F(ys)
                eys = F.split_axis(eys, batch, 0)
                h, c, ys = self.decoder(h, c, y_enc, eys)
                cys = F.concat(ys, axis=0)
                wy = self.W(cys)
                ys = self.xp.argmax(wy.data, axis=1).astype('i')
                result.append(ys)
            result = cuda.to_cpu(self.xp.concatenate(
                [self.xp.expand_dims(x, 0) for x in result]).T)
            # Remove EOS taggs
            outs = []
            for y in result:
                inds = numpy.argwhere(y == 0)
                if len(inds) > 0:
                    y = y[:inds[0, 0]]
                outs.append(y)
            return(outs)

    def beamseach(self, seq, width=3, max_length=100):
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            h, c, y_enc = self.encode([seq])
            ys = self.xp.full(1, 0, 'i')
            hyps = [(h, c, ys, 0.0, [])]
            for i in range(max_length):
                new_hyps = []
                for h, c, ys, prob, result in hyps:
                    eys = self.F(ys)
                    eys = F.split_axis(eys.data, eys.shape[0], 0)
                    h, c, ys = self.decoder(h, c, y_enc, eys)
                    cys = F.concat(ys, 0).data
                    output_prob = F.softmax(self.W(cys)).data
                    log_prob = self.xp.log(output_prob)[0]
                    log_prob[self.xp.isnan(log_prob)] = - self.xp.float32("inf")
                    ys_order = self.xp.argsort(log_prob).astype('i')[::-1]
                    for y in ys_order[:5]:
                        if len(result) > 0 and result[-1] == 0:
                            new_hyps.append(
                                (h.data, c.data, self.xp.expand_dims(y, -1), prob, result))
                        else:
                            new_hyps.append(
                                (h.data, c.data, self.xp.expand_dims(y, -1),
                                 (prob + log_prob[y]) / 6 * (
                                    5 + len(result)), result + [y.tolist()]))
                new_hyps = sorted(new_hyps, key=lambda x: x[3], reverse=True)[:5]
                hyps = new_hyps
            y = numpy.array(hyps[0][4])
            inds = numpy.argwhere(y == 0)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            return y

    def beamseach_batch(self, seq, width=3, max_length=100):
        return [self.beamseach(s, width, max_length) for s in seq]
