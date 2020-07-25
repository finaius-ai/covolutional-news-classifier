import tensorflow as tf
import pandas as pd
import json
import numpy as np

from data import weights

if __name__ == '__main__':
    # imported = tf.saved_model.load("cnn_model", tags='serve')
    #
    #
    text = "Trump Signs Executive Orders On Drug Prices"
    #
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open('vocab.json', "r").read())
    seq = tokenizer.texts_to_sequences([text])
    inputs = np.array(tf.keras.preprocessing.sequence.pad_sequences(seq, padding='post', maxlen=128))

    loaded = tf.saved_model.load('cnn_model/1')
    infer = loaded.signatures["serving_default"]
    output = infer(tf.constant(inputs, dtype=tf.float32))

    catid = np.argmax(output['dense'].numpy()[0])
    print(output['dense'].numpy()[0])
    # print(catid)

    print(list(weights.keys())[catid])

    # print(pd.Series(weights.keys()).loc[catid[:3]])
