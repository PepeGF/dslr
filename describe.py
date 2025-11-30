"""Utilities to compute and save descriptive statistics for a dataset.

This module provides the :class:`Describer` class which reads a CSV dataset,
computes common descriptive statistics for numeric columns (count, mean,
standard deviation, min, 25/50/75 percentiles, max and IQR), logs the
results and writes them to a CSV file in the same directory as the input
dataset.

Usage example::

    from pathlib import Path
    from describe import Describer

    Describer(Path("data/dataset_train.csv"))
"""

import math
import sys
from pathlib import Path

import pandas as pd

from aux_functions import input_error_handling, logger, read_data


class Describer:
    """Compute and present descriptive statistics for a dataset.

    The class reads the CSV file pointed by ``dataset_path`` and computes
    descriptive measures for all numeric columns. After construction it runs
    the full describe pipeline (calculate, print and save results).
    """

    def __init__(self, dataset_path: Path) -> None:
        """Initialize the describer and run the describe pipeline."""
        self.dataset_path: Path = dataset_path
        self.df: pd.DataFrame = read_data(self.dataset_path)
        self.describe()

    def describe(self) -> None:
        """Run the full description process.

        This method selects numeric columns and executes the sequence of
        calculation steps, then prints the results and saves them to CSV.
        """
        self.descriptions: dict = {}
        self.numeric_df: pd.DataFrame = self.df.select_dtypes(include="number")
        self.calculate_values()
        self.print_values()
        self.save_to_csv()

    def calculate_values(self) -> None:
        """Execute all individual calculation methods.

        This method calls each of the per-statistic calculation
        helpers (count, mean, std, min, percentiles, max, IQR) and populates
        the corresponding attributes on ``self``.
        """
        self.calculate_count()
        self.calculate_mean()
        self.calculate_std()
        self.calculate_minimum()
        self.calculate_percent25()
        self.calculate_percent50()
        self.calculate_percent75()
        self.calculate_maximum()
        self.calculate_interquartile_range()

    def print_values(self) -> None:
        """Assemble a DataFrame with the statistics and log it.

        The method constructs ``self.descriptions_df`` from the computed
        dictionaries and writes a human-readable representation to the
        configured logger.
        """
        self.descriptions_df: pd.DataFrame = pd.DataFrame(
            {
                "count": self.count,
                "mean": self.mean,
                "std": self.std,
                "min": self.minimum,
                "25%": self.percent25,
                "50%": self.percent50,
                "75%": self.percent75,
                "max": self.maximum,
                "IQR": self.interquartile_range,
            },
        ).transpose()
        logger.info("\n%s", self.descriptions_df)

    def save_to_csv(self) -> None:
        """Save the assembled ``descriptions_df`` to a CSV file.

        The output filename is formed by prefixing the input dataset name
        with ``descriptions_`` and writing it to the same directory as the
        input file.
        """
        output_path: Path = (
            self.dataset_path.parent / f"descriptions_{self.dataset_path.name}"
        )
        self.descriptions_df.to_csv(output_path)
        logger.info("Descriptions saved to %s", output_path)

    def calculate_count(self) -> None:
        """Compute the non-null count for each numeric column.

        The result is stored in ``self.count`` as a dictionary mapping column
        name to the number of non-NaN entries found in that column.
        """
        self.count: dict = {}
        for column in self.numeric_df.columns:
            self.count[column] = 0
            for value in self.numeric_df[column]:
                if pd.notna(value):
                    self.count[column] += 1

    def calculate_mean(self) -> None:
        """Compute the arithmetic mean for each numeric column.

        NaN values are ignored; the method uses ``self.count`` to divide the
        accumulated total and stores results in ``self.mean``.
        """
        self.mean: dict = {}
        for column in self.numeric_df.columns:
            total: float = 0
            for value in self.numeric_df[column]:
                if pd.notna(value):
                    total += value
            self.mean[column] = total / self.count[column]

    def calculate_std(self) -> None:
        """Compute the sample standard deviation for each numeric column.

        The calculation follows the textbook definition using ``self.mean``
        and dividing by ``count - 1``.
        """
        self.std: dict = {}
        for column in self.numeric_df.columns:
            total: float = 0
            for value in self.numeric_df[column]:
                if pd.notna(value):
                    total += (value - self.mean[column]) ** 2
            self.std[column] = (total / (self.count[column] - 1)) ** 0.5

    def calculate_minimum(self) -> None:
        """Find the minimum value (ignoring NaNs) for each numeric column.

        Results are stored in ``self.minimum`` as a mapping column -> minimum.
        """
        self.minimum: dict = {}
        for column in self.numeric_df.columns:
            minimum: float = self.numeric_df[column].iloc[0]
            for value in self.numeric_df[column]:
                if pd.notna(value):
                    minimum = minimum if value > minimum else value
            self.minimum[column] = minimum

    def calculate_percent25(self) -> None:
        """Compute the 25th percentile for each numeric column.

        The implementation sorts the non-NaN values and uses the average of
        the floor and ceil indexed values for the requested quantile.
        Results are stored in ``self.percent25``.
        """
        self.percent25: dict = {}
        for column in self.numeric_df.columns:
            sorted_values: pd.Series[float] = self.numeric_df[column].sort_values()
            self.percent25[column] = (
                sorted_values.iloc[math.floor(0.25 * (self.count[column] - 1))]
                + sorted_values.iloc[math.ceil(0.25 * (self.count[column] - 1))]
            ) / 2

    def calculate_percent50(self) -> None:
        """Compute the 50th percentile (median) for each numeric column.

        Uses the same index-averaging approach as other percentile helpers and
        stores results in ``self.percent50``.
        """
        self.percent50: dict = {}
        for column in self.numeric_df.columns:
            sorted_values: pd.Series[float] = self.numeric_df[column].sort_values()
            self.percent50[column] = (
                sorted_values.iloc[math.floor(0.50 * (self.count[column] - 1))]
                + sorted_values.iloc[math.ceil(0.50 * (self.count[column] - 1))]
            ) / 2

    def calculate_percent75(self) -> None:
        """Compute the 75th percentile for each numeric column.

        The value is stored in ``self.percent75`` using the same averaging
        between floor and ceil indices as the other percentile methods.
        """
        self.percent75: dict = {}
        for column in self.numeric_df.columns:
            sorted_values: pd.Series[float] = self.numeric_df[column].sort_values()
            self.percent75[column] = (
                sorted_values.iloc[math.floor(0.75 * (self.count[column] - 1))]
                + sorted_values.iloc[math.ceil(0.75 * (self.count[column] - 1))]
            ) / 2

    def calculate_maximum(self) -> None:
        """Find the maximum value (ignoring NaNs) for each numeric column.

        Results are stored in ``self.maximum`` as a mapping column -> maximum.
        """
        self.maximum: dict = {}
        for column in self.numeric_df.columns:
            maximum: float = self.numeric_df[column].iloc[0]
            for value in self.numeric_df[column]:
                if pd.notna(value):
                    maximum = maximum if value < maximum else value
            self.maximum[column] = maximum

    def read_data(self) -> pd.DataFrame:
        """Read the CSV file at ``self.dataset_path`` and return a DataFrame.

        On FileNotFoundError the function prints the exception and exits the
        program with a non-zero status code.
        """
        try:
            return pd.read_csv(self.dataset_path)
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)

    def calculate_interquartile_range(self) -> None:
        """Compute the interquartile range (IQR) per numeric column.

        The IQR is computed as Q3 - Q1 using previously calculated
        ``self.percent75`` and ``self.percent25`` dictionaries.
        """
        self.interquartile_range: dict = {}
        for column in self.numeric_df.columns:
            self.interquartile_range[column] = (
                self.percent75[column] - self.percent25[column]
            )


def main() -> None:
    """Command-line entry point for the describer utility.

    Expects a single command-line argument with the dataset filename (relative
    to the script's ``data/`` directory). Validates input using
    ``input_error_handling()`` and constructs the full path passed to
    :class:`Describer`.
    """
    input_error_handling()
    dslr_path: Path = Path(__file__).parent
    dataset_path: Path = dslr_path / f"data/{sys.argv[1]}"
    Describer(dataset_path)


if __name__ == "__main__":
    main()
