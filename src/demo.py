from tensorflow.keras.models import load_model

import joblib

from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import *


def predict_sentence(

    sentence

):

    tokenizer = joblib.load(

      TOKENIZER_PATH

    )

    model = load_model(

      MODEL_PATH

    )

    seq = tokenizer.texts_to_sequences(

      [sentence]

    )

    seq = pad_sequences(

      seq,

      maxlen=MAX_LENGTH

    )

    pred = model.predict(

      seq

    ).argmax()

    labels=[

    "BACKGROUND",

    "CONCLUSIONS",

    "METHODS",

    "OBJECTIVE",

    "RESULTS"

    ]

    print(

      labels[pred]

    )