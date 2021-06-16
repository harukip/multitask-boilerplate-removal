import tensorflow as tf
import requests
import json
import pandas as pd
import numpy as np
import re
import os


def limit_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            #tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

def preprocess_df(df, model, WORD, depth=False):
    df_tag = [str(t) for t in list(df['tag'])]
    tag_map = load_tokenizer()
    tag_map.num_words = len(tag_map.word_index)
    tag_lists = tag_map.texts_to_sequences(list(df['tag']))
    tag_tokens = tf.keras.preprocessing.sequence.pad_sequences(
        tag_lists, maxlen=50
    )

    df_content = [re.sub("\d+", "NUMPLACE", str(c))
                  for c in list(df['content'])]

    if WORD:
        word_emb = None
        word_map = load_tokenizer("word_tokenizer.json")
        for content in df_content:
            word_dict = tf.keras.preprocessing.text.Tokenizer()
            word_dict.fit_on_texts([content])
            word_vec = np.zeros(word_map.num_words, dtype=np.int32)
            for word, count in dict(word_dict.word_counts).items():
                index = word_map.word_index.get(word, 1)
                index = index if index <= word_map.num_words else 1
                word_vec[index-1] += count
            word_emb = concatAxisZero(word_emb, np.expand_dims(word_vec, 0))
        content_emb = word_emb
    else:
        bert_emb = model.bert.encode(df_content)
        content_emb = bert_emb
    if "label" not in df.columns:
        df['label'] = [-1 for _ in range(len(df))]
    label = tf.one_hot(np.array(df['label']), 2)
    if depth:
        if "depth" not in df.columns:
            df['depth'] = [len(list(filter(None, t.split(" "))))
                           for t in df.tag]
        depth = np.expand_dims(np.array(df['depth']), [-1])
        return tag_tokens, content_emb, label, depth
    return tag_tokens, content_emb, label


def get_data(file, model, WORD=False, depth=False):
    df = pd.read_csv(file)
    if depth:
        tag_out, emb_out, label_out, depth_out = preprocess_df(
            df, model, WORD, depth)
        return tag_out, emb_out, label_out, depth_out
    tag_out, emb_out, label_out = preprocess_df(df, model, WORD, depth)
    return tag_out, emb_out, label_out


def concatAxisZero(all_pred, pred):
    if all_pred is None:
        all_pred = pred
    else:
        all_pred = np.concatenate([all_pred, pred], axis=0)
    return all_pred


def load_tokenizer(name="tag_tokenizer.json"):
    #     fileName = "word_tokenizer.json" if not tag else "tag_tokenizer.json"
    with open(name, "r") as file:
        tk_json = file.read()
    return tf.keras.preprocessing.text.tokenizer_from_json(tk_json)
