import tensorflow as tf
import pandas as pd
import numpy as np
from collections import OrderedDict

weights = OrderedDict({'POLITICS': 0.16299980582814297,
                       'WELLNESS': 0.08875645372486346,
                       'ENTERTAINMENT': 0.07994901744061578,
                       'TRAVEL': 0.04922505513982863,
                       'STYLE & BEAUTY': 0.04804010893539056,
                       'PARENTING': 0.043200748806340956,
                       'HEALTHY LIVING': 0.03332785669121198,
                       'QUEER VOICES': 0.031435925776562956,
                       'FOOD & DRINK': 0.030997794406854764,
                       'BUSINESS': 0.029558931158608536,
                       'COMEDY': 0.025765111798180758,
                       'SPORTS': 0.0243162910188048,
                       'BLACK VOICES': 0.022543850477712554,
                       'HOME & LIVING': 0.02088592154461223,
                       'PARENTS': 0.019691017809044427,
                       'THE WORLDPOST': 0.018242197029668464,
                       'WEDDINGS': 0.01817747307732521,
                       'WOMEN': 0.017375891821381807,
                       'IMPACT': 0.017221550088870965,
                       'DIVORCE': 0.017057250825230394,
                       'CRIME': 0.01695269674836821,
                       'MEDIA': 0.014015225065097359,
                       'WEIRD NEWS': 0.013293304058191811,
                       'GREEN': 0.013054323311078251,
                       'WORLDPOST': 0.01284023639178902,
                       'RELIGION': 0.012725724783797106,
                       'STYLE': 0.01122213758320762,
                       'SCIENCE': 0.010843751400277815,
                       'WORLD NEWS': 0.010838772634712949,
                       'TASTE': 0.010435492623958816,
                       'TECH': 0.010365789906050693,
                       'MONEY': 0.008498752819226001,
                       'ARTS': 0.007512957237382563,
                       'FIFTY': 0.006975250556377052,
                       'GOOD NEWS': 0.006960314259682454,
                       'ARTS & CULTURE': 0.006666567091355369,
                       'ENVIRONMENT': 0.006586906842317516,
                       'COLLEGE': 0.005695707806206529,
                       'LATINO VOICES': 0.005621026322733542,
                       'CULTURE & ARTS': 0.005128128531811823,
                       'EDUCATION': 0.00499868062712531}.items()
                      )


def tokenize_dataset(df):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=30000,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(df['headline'].values)
    train_seqs = tokenizer.texts_to_sequences(df["headline"].values)
    #
    train_vectors = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post', maxlen=128)
    categories = pd.Categorical(np.squeeze(df['category'])).categories
    target = pd.Categorical(np.squeeze(df['category']), categories=categories)
    target = target.map(lambda c: {k: i for (i, k) in enumerate(weights.keys())}[c])

    new_df = pd.DataFrame(train_vectors)
    new_df['target'] = target
    new_df = new_df.sample(frac=1)

    return new_df, tokenizer
