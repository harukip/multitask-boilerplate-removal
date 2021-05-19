import tensorflow as tf
from glob import glob
from utils import util


class DataLoader():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.train_ds = tf.data.Dataset.from_generator(
            self.gen_data,
            args=[0],
            output_types=(tf.float32,
                          tf.float32,
                          tf.float32,
                          tf.float32)).padded_batch(
            batch_size=args.batch,
            padded_shapes=([None, None], [None, None], [None, args.label_size], [None, 1]),
            padding_values=(
                tf.constant(0, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32),
                tf.constant(0, dtype=tf.float32))).shuffle(buffer_size=1000)

        self.val_ds = tf.data.Dataset.from_generator(
            self.gen_data,
            args=[1],
            output_types=(tf.float32,
                          tf.float32,
                          tf.float32)).batch(1)

        self.test_ds = tf.data.Dataset.from_generator(
            self.gen_data,
            args=[2],
            output_types=(tf.float32,
                          tf.float32,
                          tf.float32)).batch(1)

    def gen_data(self, file_type):
        if file_type == 0:
            files = sorted(glob(self.args.train_folder + "*.csv"))
            for f in files:
                tag, emb, label, depth = util.get_data(f,
                                                       self.model,
                                                       self.args.word,
                                                       True)
                yield tag, emb, label, depth
        else:
            if file_type == 1:
                files = sorted(glob(self.args.val_folder + "*.csv"))
            else:
                files = sorted(glob(self.args.test_folder + "*.csv"))
            for f in files:
                tag, emb, label = util.get_data(f,
                                                self.model,
                                                self.args.word,
                                                False)
                yield tag, emb, label
