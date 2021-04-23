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


def preprocess_df(df, dataloader, WORD, depth=False):
    df_tag = [str(t) for t in list(df['tag'])]
    tag_emb = None
    tag_map = dataloader.tokenizer
    for tag in df_tag:
        tag_dict = tf.keras.preprocessing.text.Tokenizer()
        tag_dict.fit_on_texts([tag])
        tag_vec = np.zeros(tag_map.num_words, dtype=np.int32)
        for tag, count in dict(tag_dict.word_counts).items():
            index = tag_map.word_index.get(tag, 1)
            index = index if index <= tag_map.num_words else 1
            tag_vec[index-1] += count
        tag_emb = concatAxisZero(tag_emb, np.expand_dims(tag_vec, 0))

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
        bert_emb = dataloader.model.bert.encode(df_content)
        content_emb = bert_emb
    if "label" not in df.columns:
        df['label'] = [-1 for _ in range(len(df))]
    label = tf.one_hot(np.array(df['label']), 2)
    if depth:
        if "depth" not in df.columns:
            df['depth'] = [len(list(filter(None, t.split(" "))))
                           for t in df.tag]
        df['prev_x'] = df.x.shift(1, fill_value=0)
        df['next_x'] = df.x.shift(-1, fill_value=0)
        df['prev_y'] = df.y.shift(1, fill_value=0)
        df['next_y'] = df.y.shift(-1, fill_value=0)
        df['prev_width'] = df.width.shift(1, fill_value=0)
        df['next_width'] = df.width.shift(-1, fill_value=0)
        df['prev_height'] = df.height.shift(1, fill_value=0)
        df['next_height'] = df.height.shift(-1, fill_value=0)
        cols = [
            "prev_x",
            "x",
            "next_x",
            "prev_y",
            "y",
            "next_y",
            "prev_width",
            "width",
            "next_width",
            "prev_height",
            "height",
            "next_height"
        ]
        depth = np.array(df[cols])
        depth = dataloader.scaler.fit_transform(depth)
        return tag_emb, content_emb, label, depth
    return tag_emb, content_emb, label


def get_data(file, dataloader, WORD=False, depth=False):
    df = pd.read_csv(file)
    if depth:
        tag_out, emb_out, label_out, depth_out = preprocess_df(
            df, dataloader, WORD, depth)
        return tag_out, emb_out, label_out, depth_out
    tag_out, emb_out, label_out = preprocess_df(df, dataloader, WORD, depth)
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
