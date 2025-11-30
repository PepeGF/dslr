"""Generate and display confusion matrix for model predictions."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_predicted_data() -> pd.DataFrame:
    """Read predicted data from CSV file."""
    return pd.read_csv("data/houses.csv")


def read_true_data() -> pd.DataFrame:
    """Read true data from CSV file."""
    return pd.read_csv("data/dataset_truth.csv")


def generate_confusion_matrix(
    true_data: pd.DataFrame, predicted_data: pd.DataFrame
) -> np.ndarray:
    """Generate confusion matrix from true and predicted data."""
    true_labels = true_data["Hogwarts House"]
    predicted_labels = predicted_data["Hogwarts House"]
    labels = sorted(true_labels.unique())
    confusion_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    label_to_index = {label: index for index, label in enumerate(labels)}

    for true, pred in zip(true_labels, predicted_labels, strict=True):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        confusion_matrix[true_index][pred_index] += 1

    return confusion_matrix


def display_confusion_matrix(confusion_matrix: np.ndarray) -> None:
    """Display the confusion matrix using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(confusion_matrix.shape[0])
    plt.xticks(
        tick_marks,
        ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
        rotation=90,
    )
    plt.yticks(
        tick_marks,
        ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
    )

    thresh = confusion_matrix.max() / 2.0
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(
                j,
                i,
                format(confusion_matrix[i, j], "d"),
                horizontalalignment="center",
                color="white" if confusion_matrix[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()


def calculate_precision(confusion_matrix: np.ndarray) -> float:
    """Calculate precision from the confusion matrix."""
    true_positives = np.diag(confusion_matrix)
    predicted_positives = confusion_matrix.sum(axis=0)
    precision_per_class = true_positives / predicted_positives
    for precision, house in zip(
        precision_per_class,
        ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"],
        strict=True,
    ):
        print(f"Precision for {house}: {precision * 100:.2f}%")
    overall_precision = np.nanmean(precision_per_class) * 100
    print(f"Overall Precision: {overall_precision:.2f}%", end="\n\n\n")
    return overall_precision


def calculate_recall(confusion_matrix: np.ndarray) -> float:
    """Calculate recall from the confusion matrix."""
    true_positives = np.diag(confusion_matrix)
    actual_positives = confusion_matrix.sum(axis=1)
    recall_per_class = true_positives / actual_positives
    for recall, house in zip(
        recall_per_class, ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]
    ):
        print(f"Recall for {house}: {recall * 100:.2f}%")
    overall_recall = np.nanmean(recall_per_class) * 100
    print(f"Overall Recall: {overall_recall:.2f}%", end="\n\n\n")
    return overall_recall


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall."""
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"F1 Score: {f1_score:.2f}%")
    return f1_score


def main():
    """Main function to generate and display confusion matrix."""
    print("Generating Confusion Matrix and Metrics...\n")
    predicted_data: pd.DataFrame = read_predicted_data()
    true_data: pd.DataFrame = read_true_data()
    confusion_matrix = generate_confusion_matrix(true_data, predicted_data)
    precision = calculate_precision(confusion_matrix)
    recall = calculate_recall(confusion_matrix)
    f1_score = calculate_f1_score(precision, recall)
    display_confusion_matrix(confusion_matrix)


if __name__ == "__main__":
    main()
