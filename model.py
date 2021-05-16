import tensorflow as tf
from module import bertencoder
from module import tag2vec
import numpy as np


class LSTMModel(tf.keras.Model):
    def __init__(self,
                 ff_dim,
                 num_layers,
                 out_dim,
                 lr,
                 lstm_dropout,
                 dropout,
                 mc_step,
                 bert_trainable=False,
                 topk=None,
                 masking=None):
        super(LSTMModel, self).__init__()
        self.bert = bertencoder.BertEncoder(trainable=bert_trainable)
        self.tag_encoder = tag2vec.Leafnode_Encoder()
        self.topEmb_layer = tf.keras.layers.Dense(
            ff_dim, activation='relu', name="topEmb Layer")
        self.dropout_layer = tf.keras.layers.Dropout(dropout)
        self.concat_layer = tf.keras.layers.Concatenate()
        self.mask_layer = tf.keras.layers.Masking(name="masking Layer")
        self.depth_out = tf.keras.layers.Dense(1)
        self.lstms = [tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(ff_dim//2, return_sequences=True, dropout=lstm_dropout))
            for _ in range(num_layers)]
        self.label_out = tf.keras.layers.Dense(out_dim)
        self.mc_step = mc_step
        self.Opt = tf.keras.optimizers.Adam(lr)

    def call(self, t, e, training=False):
        t = self.tag_encoder(t)
        e = self.topEmb_layer(e)
        x = self.concat_layer([t, e])
        x = self.mask_layer(x)
        mask = x._keras_mask
        d = self.depth_out(t)
        for lstm in self.lstms:
            x = lstm(x, mask=mask)
        x = self.dropout_layer(x, training=training)
        lstm_out = x
        out = self.label_out(lstm_out)
        p = tf.nn.softmax(out)
        return p, lstm_out, out, d

    def MC_sampling(self, t, e, training=False):
        return self.call(t, e, training=training)


class MCModel(tf.keras.Model):
    def __init__(self,
                 ff_dim,
                 num_layers,
                 out_dim,
                 lr,
                 lstm_dropout,
                 dropout,
                 mc_step,
                 bert_trainable=False,
                 topk=None,
                 masking=None):
        super(MCModel, self).__init__()
        self.bert = bertencoder.BertEncoder(trainable=bert_trainable)
        self.tag_encoder = tag2vec.Leafnode_Encoder()
        self.topTag_layer = tf.keras.layers.Dense(
            ff_dim, activation='relu', name="topTag Layer")
        self.topEmb_layer = tf.keras.layers.Dense(
            ff_dim, activation='relu', name="topEmb Layer")
        self.concat_layer = tf.keras.layers.Concatenate()
        self.mask_layer = tf.keras.layers.Masking(name="masking Layer")
        self.depth_out = tf.keras.layers.Dense(1)
        self.lstms = [tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(ff_dim//2, return_sequences=True))
            for _ in range(num_layers)]
        self.label_out = tf.keras.layers.Dense(out_dim)
        self.mc_step = mc_step
        self.lstm_dropout = lstm_dropout
        self.dropout = dropout
        self.Opt = tf.keras.optimizers.Adam(lr)

    def add_dropout(self, x, dropout):
        ''' x: batch * seq_len * hidden '''
        return tf.keras.layers.SpatialDropout1D(rate=dropout)(x, training=True)

    def call(self, t, e):
        t = self.tag_encoder(t)
        t = self.topTag_layer(t)
        e = self.topEmb_layer(e)
        x = self.concat_layer([t, e])
        x = self.mask_layer(x)
        mask = x._keras_mask
        d = self.depth_out(t)
        for lstm in self.lstms:
            x = self.add_dropout(x, self.lstm_dropout)
            x = lstm(x, mask=mask)
        x = self.add_dropout(x, self.dropout)
        lstm_out = x
        out = self.label_out(lstm_out)
        p = tf.nn.softmax(out)
        return p, lstm_out, out, d

    def MC_sampling(self, t, e, training=False):
        mc_t = tf.repeat(t, repeats=self.mc_step, axis=0)
        mc_e = tf.repeat(e, repeats=self.mc_step, axis=0)
        p, lstm_out, out, d = self.call(mc_t, mc_e)
        p = tf.reshape(
            p, [
                t.shape[0],
                self.mc_step,
                p.shape[1],
                p.shape[2]])
        lstm_out = tf.reshape(
            lstm_out, [
                t.shape[0],
                self.mc_step,
                lstm_out.shape[1],
                lstm_out.shape[2]])
        out = tf.reshape(
            out, [
                t.shape[0],
                self.mc_step,
                out.shape[1],
                out.shape[2]])
        d = tf.reshape(
            d, [
                t.shape[0],
                self.mc_step,
                d.shape[1],
                d.shape[2]])
        p = tf.reduce_mean(p, axis=1)
        lstm_out = tf.reduce_mean(lstm_out, axis=1)
        out = tf.reduce_mean(out, axis=1)
        d = tf.reduce_mean(d, axis=1)
        return p, lstm_out, out, d
