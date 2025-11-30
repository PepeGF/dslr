"""Logistic regression training with softmax loss."""

import sys

import numpy as np
import pandas as pd

FEATURES = 13
CLASSES = 4


class SoftmaxTrainer:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.set_initial_parameters(learning_rate, epochs)
        self.read_training_data()
        self.create_binary_houses_columns()
        self.set_courses()
        self.fill_nulls()
        self.data: pd.DataFrame = self.data.drop(columns=["Index"])
        self.set_x_train()
        self.x_train_normalized: np.ndarray = self.normalize_data()
        self.set_y_train()
        self.set_model_initial_parameters()
        # logits = self.logits()
        # y_pred = self.softmax(logits)
        # loss = self.cross_entropy_loss(y_pred)
        self.train_model()
        self.save_final_weights_and_bias()

    def set_initial_parameters(self, learning_rate: float, n_iterations: int) -> None:
        """Set initial parameters for the model."""
        self.learning_rate: float = learning_rate
        self.n_iterations: int = n_iterations
        self.weights: np.ndarray = np.zeros((FEATURES, CLASSES))
        self.bias: np.ndarray = np.zeros(CLASSES)
        self.hogwarts_houses: list[str] = [
            "Gryffindor",
            "Ravenclaw",
            "Hufflepuff",
            "Slytherin",
        ]

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
        """Set the training labels for each house."""
        self.y_train = np.zeros(
            (self.x_train.shape[0], len(self.hogwarts_houses)),
            dtype=int,
        )
        for idx, house in enumerate(self.hogwarts_houses):
            self.y_train[:, idx] = self.data[f"es_{house}"].to_numpy()

    def set_model_initial_parameters(self) -> None:
        """Set model initial parameters."""
        self.m: int
        self.n: int
        self.m, self.n = self.x_train_normalized.shape
        self.final_all_w: dict[str, np.ndarray] = {}
        self.final_all_b: dict[str, np.float64] = {}
        self.predictions: dict[str, np.ndarray] = {}

    def logits(self) -> np.ndarray:
        """Compute the logits. Z = XW + b."""
        return (self.x_train_normalized @ self.weights) + self.bias  # 1600 x 4

    def softmax(self, z: np.ndarray) -> np.ndarray:  # ^Yij (probabilidad calculada)
        """Compute the softmax of each row of the input array." Y = e^z / sum(e^z).

        Se resta el z max para evitar overflow. Matemáticamente no afecta al resultado,
        pero hace que los valores no sean tan grandes y no se produzca overflow.
        """
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def cross_entropy_loss(
        self,
        y_pred: np.ndarray,
    ) -> float:
        """Compute the cross-entropy loss."""
        return -np.sum(self.y_train * np.log(y_pred + 1e-15)) / self.m

    def gradient_descent(self) -> None:
        """Compute the gradient descent."""
        dw = (
            self.x_train_normalized.T
            @ (self.softmax(self.logits()) - self.y_train)
            / self.m
        )
        db = np.sum(self.softmax(self.logits()) - self.y_train, axis=0) / self.m
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

    def train_model(self) -> None:
        """Train the model using gradient descent."""
        for i in range(self.n_iterations):
            self.gradient_descent()
            if i % 100 == 0:
                logits: np.ndarray = self.logits()
                y_pred: np.ndarray = self.softmax(logits)
                loss = self.cross_entropy_loss(y_pred)
                print(f"Iteration {i}, loss: {loss}")
        np.savetxt(
            "softmax_weights.txt",
            self.weights,
        )
        np.savetxt(
            "softmax_bias.txt",
            self.bias,
        )

    def save_final_weights_and_bias(self) -> None:
        """Save the final weights and bias to a sigle CSV file."""
        weights_bias: np.ndarray = np.vstack(
            (self.weights, self.bias.reshape(1, -1)),
        )
        weights_bias_df: pd.DataFrame = pd.DataFrame(
            weights_bias,
            columns=[
                "Gryffindor",
                "Ravenclaw",
                "Hufflepuff",
                "Slytherin",
            ],
        )
        weights_bias_df.index = [*self.courses, "bias"]
        weights_bias_df.to_csv(
            "softmax_weights_bias.csv",
            index=True,
            index_label="Courses",
        )


if __name__ == "__main__":
    # Example usage
    trainer = SoftmaxTrainer(learning_rate=0.01, epochs=4000)
