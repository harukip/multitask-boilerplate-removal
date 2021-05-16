import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, TimeDistributed

class Leafnode_Encoder(Model):
    def __init__(self):
        super(Leafnode_Encoder, self).__init__()
        self.embedding = Embedding(50, 10)
        self.lstm = Bidirectional(LSTM(32))
    
    def call(self, node):
        # node (None, None, 50)
        # embedding (None, None, 50, 10)
        # lstm (None, None, 64)
        embedding = self.embedding(node)
        return TimeDistributed(self.lstm)(embedding)