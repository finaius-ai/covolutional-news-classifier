from sklearn import datasets
import numpy as np
import pandas as pd
import tensorflow as tf

from cnn_model import cnn_model
from data import tokenize_dataset, weights


def train():
    raw_data = pd.read_json('data/compressed_News_Category_Dataset_v2.json.zip', compression='zip', lines=True)

    data, tokenizer = tokenize_dataset(raw_data)
    # vocab = open('vocab.json', "w")
    # vocab.write(tokenizer.to_json())
    Y = data.pop('target').values
    X = data.values

    num_valid = 5000
    dataset = tf.data.Dataset.from_tensor_slices((X[:-num_valid], Y[:-num_valid])).shuffle(1000).batch(32).repeat()
    validate_dataset = tf.data.Dataset.from_tensor_slices((X[-num_valid:], Y[-num_valid:])).shuffle(1000).batch(32)

    model = cnn_model(30000)

    model.fit(dataset, validation_data=validate_dataset, epochs=20, steps_per_epoch=1000, class_weight={i:k for (i, k) in enumerate(weights.values())})


    return model


if __name__ == '__main__':
    import os
    model = train()
    mobilenet_save_path = os.path.join("", "cnn_model/1/")
    tf.saved_model.save(model, mobilenet_save_path)
