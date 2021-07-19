import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed

class Leafnode_Encoder(Model):
    def __init__(self, emb_init):
        super(Leafnode_Encoder, self).__init__()
        if emb_init == 0:
            self.embedding = Embedding(
                196,
                32)
        else:
            if emb_init == 1:
                embedding_matrix = np.load("./cbow.npz")["emb"]
            else:
                embedding_matrix = np.load("./skip-gram.npz")["emb"]
            self.embedding = Embedding(
                196,
                64,
                embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))
        self.lstm = Bidirectional(LSTM(32))
    
    def call(self, node):
        # node (None, None, 50)
        # embedding (None, None, 50, 10)
        # lstm (None, None, 64)
        embedding = self.embedding(node)
        return TimeDistributed(self.lstm)(embedding)