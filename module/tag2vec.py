import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed

class Leafnode_Encoder(Model):
    def __init__(self):
        super(Leafnode_Encoder, self).__init__()
        embedding_matrix = np.load("../tag_emb/cbow.npz")["emb"][:50]
        self.embedding = Embedding(
            50,
            64,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))
        self.lstm = Bidirectional(LSTM(32))
    
    def call(self, node):
        # node (None, None, 50)
        # embedding (None, None, 50, 10)
        # lstm (None, None, 64)
        embedding = self.embedding(node)
        return TimeDistributed(self.lstm)(embedding)