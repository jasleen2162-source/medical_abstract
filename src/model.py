from tensorflow.keras import Sequential

from tensorflow.keras.layers import *

from src.config import *


def build_model():

    model = Sequential([

      Embedding(

      MAX_WORDS,

      128,

      input_length=MAX_LENGTH

      ),

      Bidirectional(

      LSTM(

      64

      )

      ),

      Dropout(

      0.5

      ),

      Dense(

      64,

      activation="relu"

      ),

      Dense(

      5,

      activation="softmax"

      )

    ])

    model.compile(

      optimizer="adam",

      loss="sparse_categorical_crossentropy",

      metrics=["accuracy"]

    )

    return model