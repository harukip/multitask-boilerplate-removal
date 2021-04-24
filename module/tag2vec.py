import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dot, Embedding, Flatten, Dense, LSTM


class SkipGram(Model):
    def __init__(self, vocab_size, embedding_dim, num_ns):
        super(SkipGram, self).__init__()
        self.target_embedding = Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding")
        self.context_embedding = Embedding(vocab_size,
                                           embedding_dim,
                                           input_length=num_ns+1)
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        word_emb = self.target_embedding(target)
        context_emb = self.context_embedding(context)
        dots = self.dots([context_emb, word_emb])
        return self.flatten(dots)

class CBOW(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.context_embedding = Embedding(vocab_size,
                                          embedding_dim,
                                          input_length=1,
                                          name="w2v_embedding")
        self.out = Dense(vocab_size+1)

    def call(self, context):
        context_emb = self.context_embedding(context)
        context_sum = tf.math.reduce_sum(context_emb, axis=1)
        return self.out(context_sum)

class Leafnode_Encoder(Model):
    def __init__(self):
        super(Leafnode_Encoder, self).__init__()
        self.low_level_lstm = LSTM(128, return_sequences=True)
        self.high_level_lstm = LSTM(128)
        self.mlp = Dense(128, activation='relu')
        self.predict_layer = Dense(1, activation='sigmoid')
    
    def call(self, node):
        low_level_feature = self.low_level_lstm(node)
        high_level_feature = self.high_level_lstm(low_level_feature)
        mlp = self.mlp(high_level_feature)
        return self.predict_layer(mlp)

    def get_embedding(self, node):
        low_level_feature = self.low_level_lstm(node)
        return self.high_level_lstm(low_level_feature)