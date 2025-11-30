""""""

import sys
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import (
    ndarray,
)
from pandas.core.frame import DataFrame

if typing.TYPE_CHECKING:
    from matplotlib.figure import Figure
    from pandas.core.groupby.generic import DataFrameGroupBy
    from pandas.core.indexes.base import Index

from aux_functions import input_error_handling, read_data


def main(df: DataFrame) -> None:
    """"""
    # Create a histogram for each numeric column
    numeric_columns: Index[str] = df.select_dtypes(include="number").columns.drop(
        "Index",
        errors="raise",
    )
    grouped_data: DataFrameGroupBy = df.groupby("Hogwarts House")

    house_colors: dict[str, str] = {
        "Gryffindor": "red",
        "Hufflepuff": "orange",
        "Ravenclaw": "blue",
        "Slytherin": "green",
    }

    for column in numeric_columns:
        data: list[ndarray[tuple[int]]] = [
            group[column].dropna().to_numpy() for _, group in grouped_data
        ]
        labels: list[str] = grouped_data.groups.keys()
        colors: list[str] = [house_colors[label] for label in labels]
        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"Histogram of {column}")
        fig.canvas.manager.window.wm_geometry("+100+100")
        plt.hist(data, color=colors, label=labels, bins=20, alpha=0.7)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend(title="Hogwarts House")
        plt.show(block=False)
        plt.pause(0.2)
        # plt.clf()
        plt.close()

    num_columns: int = len(numeric_columns)
    rows: int = (num_columns + 3) // 5
    fig, axes_tmp = plt.subplots(rows, 5, figsize=(15, 3 * rows))
    axes = axes_tmp.flatten()
    for i, column in enumerate(numeric_columns):
        ax = axes[i]
        data: list[ndarray[tuple[int]]] = [
            group[column].dropna().to_numpy() for _, group in grouped_data
        ]
        labels: list[str] = grouped_data.groups.keys()
        colors: list[str] = [house_colors[label] for label in labels]
        ax.hist(data, color=colors, label=labels, bins=20, alpha=0.7)
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        ax.legend(title="Hogwarts House", fontsize="small")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.canvas.manager.set_window_title(f"Histograms of Numeric Features")
    fig.canvas.manager.window.wm_geometry("+50+50")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    input_error_handling()
    dslr_path: Path = Path(__file__).parent
    dataset_path: Path = dslr_path / f"data/{sys.argv[1]}"
    df: DataFrame = read_data(dataset_path)
    main(df)
