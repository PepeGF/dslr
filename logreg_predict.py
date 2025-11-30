"""Logistic regression prediction module for DSLR model.

Functions to predict the Hogwarts House using trained weights and bias.
"""

import sys

import numpy as np
import pandas as pd


class DslrPredict:
    """Logistic regression prediction module for DSLR model."""

    def __init__(self) -> None:
        """Initialize the prediction module."""
        self.read_prediction_data()
        self.prepare_dataset_for_prediction()
        self.read_model_parameters()
        self.prepare_parameters()
        self.calculate_predictions()
        self.choose_house()
        self.save_predictions_file()

    def save_predictions_file(self) -> None:
        """Save the predicted houses to a CSV file."""
        self.predicted_houses.to_csv(
            "data/houses.csv",
            index=True,
            index_label="Index",
            header=["Hogwarts House"],
        )
        print("Hogwarts Houses predicted successfully to data/houses.csv")

    def choose_house(self) -> None:
        """Choose the house with the highest prediction probability."""
        self.predicted_houses: pd.Series[int] = self.predictions_df.idxmax(axis=1)

    def prepare_dataset_for_prediction(self) -> None:
        """Prepare the dataset for prediction."""
        self.data: pd.DataFrame = self.data_raw.drop(
            columns=[
                "Index",
                "Hogwarts House",
                "First Name",
                "Last Name",
                "Birthday",
                "Best Hand",
            ],
        )
        self.fill_nulls_with_means()
        self.normalize_data_for_prediction()
        self.x_predict: np.ndarray = self.data.to_numpy().astype(np.float64)

    def normalize_data_for_prediction(self) -> None:
        """Normalize the data for prediction using z-score normalization."""
        means: pd.Series[np.float64] = self.data.mean(axis=0)
        stds: pd.Series[np.float64] = self.data.std(axis=0)
        stds[stds == 0] = 1  # Prevent division by zero
        self.data = (self.data - means) / stds
        self.data.to_csv("data/x_predict_normalized.csv", index=False)

    def fill_nulls_with_means(self) -> None:
        """Fill null values with the mean of the column for each house."""
        numeric_columns: list[str] = self.data.select_dtypes(
            include="number",
        ).columns.to_list()
        self.data[numeric_columns] = self.data[numeric_columns].transform(
            lambda x: x.fillna(x.mean()),
        )

    def read_prediction_data(self) -> None:
        """Read the data for prediction from CSV file."""
        try:
            self.data_raw: pd.DataFrame = pd.read_csv("data/dataset_test.csv")
        except FileNotFoundError:
            try:
                self.data_raw: pd.DataFrame = pd.read_csv(
                    "dslr/dslr/data/dataset_test.csv"
                )
            except FileNotFoundError:
                print("Prediction data file not found in both paths.")
                sys.exit(1)

    def prepare_parameters(self) -> None:
        """Prepare model parameters for prediction."""
        self.courses: list[str] = self.weights_and_bias.index.tolist()[:-1]
        self.w: np.ndarray = (
            self.weights_and_bias.iloc[:-1].to_numpy().astype(np.float64)
        )
        self.m: int = self.x_predict.shape[0]
        self.n: int = self.x_predict.shape[1]

    def calculate_predictions(self) -> None:
        """Calculate predictions for each Hogwarts House and save as dataframe."""
        houses: list[str] = self.weights_and_bias.columns.tolist()
        predictions: dict[str, list[float]] = {house: [] for house in houses}

        for i in range(self.m):
            x_i: np.ndarray = self.x_predict[i].reshape(self.n, 1)
            for house in houses:
                w_h: np.ndarray = self.w[:, houses.index(house)].reshape(self.n, 1)
                b_h: np.float64 = self.weights_and_bias.loc["bias", house]
                z: np.float64 = np.dot(w_h.T, x_i).item() + b_h
                a: np.float64 = self.sigmoid(z)
                predictions[house].append(a)

        self.predictions_df: pd.DataFrame = pd.DataFrame(predictions)

    def sigmoid(self, z: np.float64) -> np.float64:
        """Compute the sigmoid of z."""
        return 1 / (1 + np.exp(-z))

    def read_model_parameters(self) -> None:
        """Read the trained weights and bias from CSV file."""
        file: str = (
            "softmax_weights_bias"
            if len(sys.argv) > 1
            else "final_all_weights_and_bias"
        )
        try:
            self.weights_and_bias: pd.DataFrame = pd.read_csv(
                f"data/{file}.csv",
                index_col="Courses",
            )
        except FileNotFoundError:
            try:
                self.weights_and_bias: pd.DataFrame = pd.read_csv(
                    f"dslr/dslr/data/{file}.csv",
                    index_col="Courses",
                )
            except FileNotFoundError:
                print("Model parameters file not found in both paths.")
                sys.exit(1)
        # TODO: <Jose>: Pedir al usuario el path si no se encuentra en los dos anteriores


if __name__ == "__main__":
    DslrPredict()
