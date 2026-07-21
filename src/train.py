from src.preprocess import *

from src.model import *

from src.config import *


def train():

    (

    X_train,

    y_train,

    X_val,

    y_val,

    _,

    _,

    _

    ) = prepare_data()

    model = build_model()

    model.fit(

      X_train,

      y_train,

      validation_data=(

      X_val,

      y_val

      ),

      epochs=EPOCHS,

      batch_size=BATCH_SIZE

    )

    model.save(

      MODEL_PATH

    )