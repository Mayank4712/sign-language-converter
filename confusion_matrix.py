import os
import pickle

import matplotlib

# Use a non-interactive backend so the script can run headless and save the figure.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Core file locations and test split settings.
CSV_PATH = "landmarks_data.csv"
MODEL_PATH = os.path.join("model", "sign_model.h5")
ENCODER_PATH = os.path.join("model", "label_encoder.pkl")
TEST_SIZE = 0.2
RANDOM_STATE = 42
CONFUSION_MATRIX_PATH = "confusion_matrix.png"


def load_dataset(csv_path: str) -> pd.DataFrame:
    # Load the landmarks CSV with a helpful error if the file is missing or invalid.
    try:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        print(f"Loading dataset from {csv_path}...")
        return pd.read_csv(csv_path)
    except Exception as error:
        raise RuntimeError(f"Unable to load dataset: {error}") from error


def load_model_artifacts() -> tuple:
    # Load the trained Keras model and saved label encoder.
    try:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
    except Exception as error:
        raise RuntimeError(f"Unable to load model: {error}") from error

    try:
        if not os.path.isfile(ENCODER_PATH):
            raise FileNotFoundError(f"Label encoder not found at {ENCODER_PATH}")
        print(f"Loading label encoder from {ENCODER_PATH}...")
        with open(ENCODER_PATH, mode="rb") as file_handle:
            encoder = pickle.load(file_handle)
    except Exception as error:
        raise RuntimeError(f"Unable to load label encoder: {error}") from error

    return model, encoder


def validate_dataset(data_frame: pd.DataFrame) -> None:
    # Confirm the dataset has the label column and exactly 63 features.
    if data_frame.empty:
        raise ValueError("Landmarks dataset is empty.")
    if "label" not in data_frame.columns:
        raise ValueError("CSV must contain a 'label' column.")
    feature_frame = data_frame.drop(columns=["label"])
    if feature_frame.shape[1] != 63:
        raise ValueError(f"Expected 63 feature columns, found {feature_frame.shape[1]}.")


def plot_confusion_matrix(cm: np.ndarray, class_names: np.ndarray) -> None:
    # Render a large color heatmap and save it to disk.
    try:
        plt.figure(figsize=(20, 16))
        sns.heatmap(
            cm,
            annot=False,
            cmap="viridis",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.25,
            linecolor="white",
            cbar=True,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")
    except Exception as error:
        print(f"Warning: could not save confusion matrix image: {error}")


def print_low_accuracy_labels(cm: np.ndarray, class_names: np.ndarray) -> None:
    # Print any labels whose class-wise accuracy falls below 85 percent.
    print("Labels below 85% accuracy:")
    found_low = False
    for index, label in enumerate(class_names):
        total = int(cm[index].sum())
        correct = int(cm[index, index])
        accuracy = (correct / total) * 100.0 if total else 0.0
        if accuracy < 85.0:
            print(f"  {label}: {accuracy:.2f}%")
            found_low = True
    if not found_low:
        print("  None")


def print_top_confusions(cm: np.ndarray, class_names: np.ndarray) -> None:
    # Print the top five most confused letter pairs as percentages of each true label row.
    pairs = []
    for true_index, true_label in enumerate(class_names):
        row_total = int(cm[true_index].sum())
        if row_total == 0:
            continue
        for pred_index, pred_label in enumerate(class_names):
            if pred_index == true_index:
                continue
            count = int(cm[true_index, pred_index])
            if count == 0:
                continue
            percentage = (count / row_total) * 100.0
            pairs.append((percentage, true_label, pred_label))

    pairs.sort(key=lambda item: item[0], reverse=True)
    print("Top 5 confused letter pairs:")
    if not pairs:
        print("  None")
        return

    for percentage, true_label, pred_label in pairs[:5]:
        print(f"  {true_label} is confused with {pred_label} {percentage:.0f}% of the time")


def main() -> None:
    # Load the dataset and runtime artifacts before generating the evaluation report.
    try:
        print("Starting confusion matrix generation...")
        data_frame = load_dataset(CSV_PATH)
        validate_dataset(data_frame)
        model, encoder = load_model_artifacts()
    except Exception as error:
        print(f"Error: {error}")
        return

    labels = data_frame["label"].astype(str).values
    features = data_frame.drop(columns=["label"]).values.astype(np.float32)

    # Use the exact 20 percent split required for evaluation.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Training rows: {x_train.shape[0]}")
    print(f"Testing rows: {x_test.shape[0]}")

    # Predict the test split and convert indices back to labels with the saved encoder.
    try:
        print("Running model predictions on test split...")
        probabilities = model.predict(x_test, verbose=0)
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_labels = encoder.inverse_transform(predicted_indices)
    except Exception as error:
        print(f"Error: unable to run predictions: {error}")
        return

    class_names = np.array(encoder.classes_)
    cm = confusion_matrix(y_test, predicted_labels, labels=class_names)

    plot_confusion_matrix(cm, class_names)
    print_low_accuracy_labels(cm, class_names)
    print_top_confusions(cm, class_names)
    print("Confusion matrix analysis complete.")


if __name__ == "__main__":
    main()
