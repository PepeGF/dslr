"""Scatter plot script for training data."""

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    """Scatter plots for similar features."""
    plt.style.use("Solarize_Light2")
    df: pd.DataFrame = pd.read_csv("data/dataset_train.csv")
    colors = [
        "red"
        if x == "Gryffindor"
        else "blue"
        if x == "Ravenclaw"
        else "green"
        if x == "Slytherin"
        else "yellow"
        for x in df["Hogwarts House"]
    ]

    features: list[tuple[str, str]] = [
        ("Astronomy", "Defense Against the Dark Arts"),
        ("Ancient Runes", "Astronomy"),
        ("Arithmancy", "Care of Magical Creatures"),
    ]

    for x_feature, y_feature in features:
        plt.figure()
        plt.scatter(df[x_feature], df[y_feature], c=colors, alpha=0.5)
        plt.title(f"{y_feature} vs {x_feature}")
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.show()


if __name__ == "__main__":
    main()
