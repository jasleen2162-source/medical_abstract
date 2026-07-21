import pandas as pd

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import joblib

from src.config import *


def load_file(path):

    labels=[]

    texts=[]

    with open(

        path,

        encoding="utf-8"

    ) as f:

        for line in f:

            if "\t" not in line:

                continue

            label,text = line.strip().split(

                "\t",

                1

            )

            labels.append(

                label

            )

            texts.append(

                text

            )

    return texts,labels


def prepare_data():

    X_train,y_train = load_file(

        "data/train.txt"

    )

    X_val,y_val = load_file(

        "data/dev.txt"

    )

    X_test,y_test = load_file(

        "data/test.txt"

    )

    tokenizer = Tokenizer(

        num_words=MAX_WORDS,

        oov_token="<OOV>"

    )

    tokenizer.fit_on_texts(

        X_train

    )

    joblib.dump(

        tokenizer,

        TOKENIZER_PATH

    )

    X_train = pad_sequences(

        tokenizer.texts_to_sequences(

            X_train

        ),

        maxlen=MAX_LENGTH

    )

    X_val = pad_sequences(

        tokenizer.texts_to_sequences(

            X_val

        ),

        maxlen=MAX_LENGTH

    )

    X_test = pad_sequences(

        tokenizer.texts_to_sequences(

            X_test

        ),

        maxlen=MAX_LENGTH

    )

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(

        y_train

    )

    y_val = encoder.transform(

        y_val

    )

    y_test = encoder.transform(

        y_test

    )

    return (

        X_train,

        y_train,

        X_val,

        y_val,

        X_test,

        y_test,

        encoder

    )