import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from module import tokenization

class BertEncoder():
    
    def __init__(self, 
                 url="https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2", 
                 max_seq_length=128, 
                 num_split=32, 
                 trainable=False):
        self._max_seq_length = max_seq_length
        self._bert_layer = hub.KerasLayer(url, trainable=trainable)
        self._make_model()
        print("Bert loaded")
        vocab_file = self._bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = self._bert_layer.resolved_object.do_lower_case.numpy()
        self._tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
        self._cls_id, self._sep_id = self._tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        self.NUM_SPLIT_LENGTH = num_split
    
    def _make_model(self):
        max_seq_length = self._max_seq_length
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                               name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                           name="input_mask")
        segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
                                            name="segment_ids")
        pooled, seq = self._bert_layer([input_word_ids, input_mask, segment_ids])
        self._model = tf.keras.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled, seq])
    
    def get_model(self):
        return self._model
    
    def _convert_id(self, s, tokenizer):
        tokens = list(tokenizer.tokenize(str(s)))[:self._max_seq_length-2]
        tokens = tokenizer.convert_tokens_to_ids(tokens)
        tokens = [self._cls_id] + tokens
        tokens += [self._sep_id]
        input_mask = [1]*len(tokens)
        while(len(tokens) < self._max_seq_length):
            tokens.append(0)
            input_mask.append(0)
        return np.array(tokens).astype(np.int32), np.array(input_mask).astype(np.int32), np.array([0]*self._max_seq_length).astype(np.int32)
    
    def convert_ids(self, arrayOfSentences):
        all_ids = None
        all_masks = None
        all_segments = None
        for i in arrayOfSentences:
            ids, masks, segments = self._convert_id(i, self._tokenizer)
            if all_ids is None:
                all_ids = np.expand_dims(ids, 0)
                all_masks = np.expand_dims(masks, 0)
                all_segments = np.expand_dims(segments, 0)
            else:
                all_ids = np.concatenate([all_ids, np.expand_dims(ids, 0)], axis=0)
                all_masks = np.concatenate([all_masks, np.expand_dims(masks, 0)], axis=0)
                all_segments = np.concatenate([all_segments, np.expand_dims(segments, 0)], axis=0)
        return all_ids, all_masks, all_segments
    
    def cond(self, step, out):
        return step < len(self.SIZE_SPLIT)
    
    def body(self, step, out):
        ids = tf.gather(self.ids, step, axis=0)
        masks = tf.gather(self.masks, step, axis=0)
        segments = tf.gather(self.segments, step, axis=0)
        bert_out = self._bert_layer([ids, masks, segments])[0]
        out = out.write(step, tf.expand_dims(bert_out, 0))
        return step + 1, out
    
    def encode(self, arrayOfSentences):
        PAD = 0
        if len(arrayOfSentences)%self.NUM_SPLIT_LENGTH > 0:
            PAD = self.NUM_SPLIT_LENGTH - len(arrayOfSentences)%self.NUM_SPLIT_LENGTH
            arrayOfSentences += ["" for _ in range(PAD)]
        ids, masks, segments = self.convert_ids(arrayOfSentences)
        SIZE_SPLIT = [self.NUM_SPLIT_LENGTH for _ in range((len(arrayOfSentences)+PAD)//self.NUM_SPLIT_LENGTH)]
        self.SIZE_SPLIT = SIZE_SPLIT
        self.ids = tf.split(ids, num_or_size_splits=SIZE_SPLIT, axis=0)
        self.masks = tf.split(masks, num_or_size_splits=SIZE_SPLIT, axis=0)
        self.segments = tf.split(segments, num_or_size_splits=SIZE_SPLIT, axis=0)
        out = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        step = tf.constant(0)
        _, seq_out = tf.while_loop(self.cond, self.body, loop_vars=[step, out])
        encode_out = tf.reshape(seq_out.stack(), [-1, 768]) if not PAD else tf.reshape(seq_out.stack(), [-1, 768])[:-PAD]
        seq_out.close()
        return encode_out
