from sklearn.metrics import *

from tensorflow.keras.models import load_model

from src.preprocess import *


def evaluate_model():

    (

    _,

    _,

    _,

    _,

    X_test,

    y_test,

    encoder

    ) = prepare_data()

    model = load_model(

      MODEL_PATH

    )

    y_prob = model.predict(

      X_test

    )

    y_pred = y_prob.argmax(

      axis=1

    )

    print(

      classification_report(

      y_test,

      y_pred,

      target_names=

      encoder.classes_

      )

    )

    print(

      confusion_matrix(

      y_test,

      y_pred

      )

    )
