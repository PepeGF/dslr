"""Module for reading data from a CSV file into a pandas DataFrame."""

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger: logging.Logger = logging.getLogger(__name__)


def read_data(path: str | Path) -> pd.DataFrame:
    """Read a CSV file from the given path and returns a pandas DataFrame."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)


def input_error_handling() -> None:
    """"""
    max_args: int = 2
    if len(sys.argv) == 1:
        logger.error(
            "%s: %s",
            ValueError.__name__,
            "Please provide at least one argument.",
        )
        sys.exit(1)
    if len(sys.argv) > max_args:
        logger.error(
            "%s: %s",
            ValueError.__name__,
            "Please provide only one argument.",
        )
        sys.exit(1)
