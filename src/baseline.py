from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

import joblib


def build_baseline():

    model = Pipeline([

      (

      "tfidf",

      TfidfVectorizer()

      ),

      (

      "clf",

      LogisticRegression(

      max_iter=1000

      )

      )

    ])

    return model