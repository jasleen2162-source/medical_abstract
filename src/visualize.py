import pandas as pd

import matplotlib.pyplot as plt

from collections import Counter

from src.preprocess import load_file


def visualize_labels():

    texts, labels = load_file(

        "data/train.txt"

    )

    counts = Counter(

        labels

    )

    print()

    print("Label Counts")

    print()

    for label, count in counts.items():

        print(

            f"{label}: {count}"

        )

    plt.figure(

        figsize=(8,5)

    )

    plt.bar(

        counts.keys(),

        counts.values()

    )

    plt.xlabel(

        "Labels"

    )

    plt.ylabel(

        "Count"

    )

    plt.title(

        "Sentence Distribution"

    )

    plt.savefig(

        "reports/label_distribution.png"

    )

    plt.show()


def show_samples(

    n=5

):

    texts, labels = load_file(

        "data/train.txt"

    )

    print()

    print("Sample Data")

    print()

    for i in range(n):

        print(

            f"Label : {labels[i]}"

        )

        print(

            f"Sentence : {texts[i]}"

        )

        print("-"*50)