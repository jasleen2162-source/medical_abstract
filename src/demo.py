import re
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import MODEL_PATH, TOKENIZER_PATH, MAX_LENGTH


LABELS = [
    "BACKGROUND",
    "CONCLUSIONS",
    "METHODS",
    "OBJECTIVE",
    "RESULTS"
]


def split_into_sentences(abstract):
    """
    Split a medical abstract into individual sentences.
    """

    abstract = abstract.strip()

    if not abstract:
        return []

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9])',
        abstract
    )

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def predict_abstract(abstract):

    tokenizer = joblib.load(TOKENIZER_PATH)
    model = load_model(MODEL_PATH)

    sentences = split_into_sentences(abstract)

    if not sentences:
        return []

    sequences = tokenizer.texts_to_sequences(sentences)

    padded_sequences = pad_sequences(
        sequences,
        maxlen=MAX_LENGTH
    )

    predictions = model.predict(
        padded_sequences,
        verbose=0
    )

    predicted_classes = predictions.argmax(axis=1)

    results = []

    for sentence, prediction in zip(
        sentences,
        predicted_classes
    ):

        results.append({
            "sentence": sentence,
            "label": LABELS[prediction]
        })

    return results