"""Logistic regression model for DSLR from scratch.

One-vs-all approach for multi-class classification.
"""

import sys
import typing
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if typing.TYPE_CHECKING:
    from matplotlib.pyplot import Axes, Figure


class DslrTrain:
    """Logistic regression model for DSLR from scratch.

    One-vs-all approach for multi-class classification.
    Gradient descent optimization.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        *,
        plot: bool = False,
    ) -> None:
        """Trains the model and saves the parameters."""
        self.set_initial_parameters(learning_rate, n_iterations)
        self.read_training_data()
        self.create_binary_houses_columns()
        self.set_courses()
        self.data: pd.DataFrame = self.data.drop(columns=["Index"])
        self.fill_nulls()
        self.set_x_train()
        self.x_train_normalized: np.ndarray = self.normalize_data()
        self.set_y_train()
        self.set_model_initial_parameters()
        self.train_model()
        self.save_final_weights_and_bias()
        if plot:
            self.plot_weights_and_bias_history()

    def read_training_data(self) -> None:
        """Read the training data from CSV file."""
        try:
            self.data: pd.DataFrame = pd.read_csv("data/dataset_train.csv")
        except FileNotFoundError:
            try:
                self.data: pd.DataFrame = pd.read_csv(
                    "dslr/dslr/data/dataset_train.csv"
                )
            except FileNotFoundError:
                print("Model parameters file not found in both paths.")
                sys.exit(1)
            # TODO: Pedir al usuario el path si no se encuentra en los dos anteriores

    def set_initial_parameters(self, learning_rate: float, n_iterations: int) -> None:
        """Set initial parameters for the model."""
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.weights = None
        self.bias = None
        self.hogwarts_houses: list[str] = [
            "Gryffindor",
            "Ravenclaw",
            "Hufflepuff",
            "Slytherin",
        ]

    def create_binary_houses_columns(self) -> None:
        """Convert houses to binary classification."""
        for house in self.hogwarts_houses:
            self.data[f"es_{house}"] = (self.data["Hogwarts House"] == house).astype(
                int,
            )

    def set_courses(self) -> None:
        """Set the course list."""
        self.courses: list[str] = [
            "Arithmancy",
            "Astronomy",
            "Herbology",
            "Defense Against the Dark Arts",
            "Divination",
            "Muggle Studies",
            "Ancient Runes",
            "History of Magic",
            "Transfiguration",
            "Potions",
            "Care of Magical Creatures",
            "Charms",
            "Flying",
        ]

    def fill_nulls(self) -> None:
        """Fill null values with the mean of the column for each house."""
        numeric_columns: list[str] = self.data.select_dtypes(
            include="number"
        ).columns.to_list()
        self.data[numeric_columns] = self.data.groupby("Hogwarts House")[
            numeric_columns
        ].transform(lambda x: x.fillna(x.mean()))

    def set_x_train(self) -> None:
        """Set the training data."""
        self.x_train: np.ndarray = self.data.drop(
            columns=[
                "Hogwarts House",
                "First Name",
                "Last Name",
                "Birthday",
                "Best Hand",
                "es_Gryffindor",
                "es_Ravenclaw",
                "es_Hufflepuff",
                "es_Slytherin",
            ],
        ).to_numpy()

    def normalize_data(self) -> np.ndarray:
        """Normalize the data using z-score normalization."""
        means: np.ndarray = self.x_train.mean(axis=0)
        stds: np.ndarray = self.x_train.std(axis=0)
        stds[stds == 0] = 1  # Prevent division by zero
        return (self.x_train - means) / stds

    def set_y_train(self) -> None:
        self.y_train = {}
        for house in self.hogwarts_houses:
            self.y_train[house] = self.data[f"es_{house}"].to_numpy()

    def set_model_initial_parameters(self) -> None:
        """Set model initial parameters."""
        self.m: int
        self.n: int
        self.m, self.n = self.x_train.shape
        self.final_all_w: dict[str, np.ndarray] = {}
        self.final_all_b: dict[str, np.float64] = {}
        # self.predictions: dict[str, np.ndarray] = {}

    def train_model(self) -> None:
        """Train the model using gradient descent."""
        inicio: float = time()
        self.final_w: np.ndarray
        self.final_b: np.float64
        for house in self.hogwarts_houses:
            print(f"Training for house: {house}")
            y_train: np.ndarray = self.y_train[house]
            self.final_w, self.final_b = self.gradient_descent(
                self.x_train_normalized,
                y_train,
            )
            self.final_all_w[house] = self.final_w
            self.final_all_b[house] = self.final_b
            fin: float = time()
            print(f"{house} training time: {fin - inicio:.2f} seconds")
            print("-" * 40)
            inicio = fin

    def save_final_weights_and_bias(self) -> None:
        """Save the final weights and bias to a CSV file."""
        weights_and_bias: pd.DataFrame = pd.DataFrame(self.final_all_w)
        weights_and_bias.index = self.courses
        weights_and_bias.loc["bias"] = pd.Series(self.final_all_b)
        weights_and_bias.to_csv(
            "data/final_all_weights_and_bias.csv",
            index=True,
            index_label="Courses",
        )

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid of z."""
        return 1 / (1 + np.exp(-z))

    def cost_function(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        b: np.float64,
    ) -> np.float64:
        """Vectorized logistic cost (mean binary cross-entropy)."""
        z: np.ndarray = X @ w + b
        g_base = self.sigmoid(z)
        eps = 1e-15  # Small constant to avoid log(0)
        g: np.ndarray = np.clip(g_base, eps, 1 - eps)
        cost_arr: np.ndarray = -(y * np.log(g) + (1 - y) * np.log(1 - g)) * np.float64(
            1 / self.m
        )
        cost: np.float64 = np.sum(cost_arr)
        return np.float64(cost)

    def gradient_function(
        self,
        X: np.ndarray,
        y: np.ndarray[np.int64],
        w: np.ndarray,
        b: np.float64,
    ) -> tuple[np.float64, np.ndarray]:
        """Compute the gradient of the cost function."""
        z: np.ndarray = X @ w + b
        g: np.ndarray = self.sigmoid(z)
        error: np.ndarray = g - y  # shape (m,)
        grad_b: np.float64 = np.float64(error.mean())
        grad_w: np.ndarray = (X.T @ error) / self.m  # shape (n,)
        return grad_b, grad_w

    def gradient_descent(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.float64]:
        w: np.ndarray = np.zeros(self.n)
        b: np.float64 = 0.0
        grad_b: np.float64
        grad_w: np.ndarray
        self.w_history: list[np.ndarray] = []
        self.b_history: list[np.float64] = []
        for i in range(self.n_iterations + 1):
            grad_b, grad_w = self.gradient_function(X, y, w, b)
            b -= self.learning_rate * grad_b
            w -= self.learning_rate * grad_w
            if i % 100 == 0:
                self.w_history.append(w.copy())
                self.b_history.append(b)
                cost: np.float64 = self.cost_function(X, y, w, b)
                print(f"Iteration {i}: Cost {cost:.8f}")
        return w, b

    def plot_weights_and_bias_history(self) -> None:
        """Plot the weights and bias for each Hogwarts House."""
        houses: list[str] = self.hogwarts_houses
        n_weights: int = len(self.courses)
        n_params = n_weights + 1

        data_matrix = []
        for h in houses:
            w: np.ndarray = np.asarray(self.final_all_w.get(h))
            b = float(self.final_all_b.get(h))
            if w.shape[0] != n_weights:
                raise ValueError(
                    f"House {h} weights length mismatch: {w.shape[0]} vs {n_weights}"
                )
            data_matrix.append(np.concatenate([w, np.array([b])]))
        data_matrix: np.ndarray = np.vstack(data_matrix)
        param_labels: list[str] = [f"{name}" for name in self.courses] + ["bias"]
        house_colors: dict[str, str] = {
            "Gryffindor": "#C02E1D",
            "Ravenclaw": "#2B5EA6",
            "Hufflepuff": "#E1C22E",
            "Slytherin": "#1F8A4C",
        }
        colors: list[str | None] = [house_colors.get(h) for h in houses]
        x = np.arange(n_params)
        width: float = 0.8 / len(houses)
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(figsize=(16, 6))
        for i, h in enumerate(houses):
            offsets = x - 0.4 + i * width + width / 2
            ax.bar(offsets, data_matrix[i], width, label=h, color=colors[i])

        ax.set_xticks(x)
        ax.set_xticklabels(param_labels, rotation=45, ha="right")
        ax.set_ylabel("Parameter value")
        ax.set_title("Final parameters by Hogwarts House (weights and bias)")
        ax.legend(title="Hogwarts House")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    if sys.argv[-1] == "--plot":
        model: DslrTrain = DslrTrain(plot=True)
    else:
        model: DslrTrain = DslrTrain()
