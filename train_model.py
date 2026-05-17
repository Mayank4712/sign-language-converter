import os
import pickle

import matplotlib

# Use a non-interactive backend so training can run headless.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Core training paths and hyperparameters.
CSV_PATH = "landmarks_data_normalized.csv"
MODEL_PATH = os.path.join("model", "sign_model.h5")
ENCODER_PATH = os.path.join("model", "label_encoder.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
CONFUSION_MATRIX_PATH = "confusion_matrix.png"
TRAINING_CURVES_PATH = "training_curves.png"
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.2
RANDOM_STATE = 42
LEARNING_RATE = 0.001


def ensure_model_directory() -> None:
    # Create model directory before writing artifacts.
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    except Exception as error:
        raise RuntimeError(f"Unable to create model directory: {error}") from error


def load_dataset(csv_path: str) -> pd.DataFrame:
    # Load dataset with explicit error handling.
    try:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        print(f"Loading normalized dataset from {csv_path}...")
        return pd.read_csv(csv_path)
    except Exception as error:
        raise RuntimeError(f"Unable to load dataset: {error}") from error


def save_pickle_artifact(path: str, obj, artifact_name: str) -> None:
    # Save pickled artifact to disk.
    try:
        with open(path, mode="wb") as file_handle:
            pickle.dump(obj, file_handle)
        print(f"Saved {artifact_name} to {path}")
    except Exception as error:
        raise RuntimeError(f"Unable to save {artifact_name}: {error}") from error


def build_model(num_classes: int) -> Sequential:
    # Build deeper model architecture for stronger representation learning.
    model = Sequential(
        [
            Input(shape=(63,)),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.4),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.2),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(train_integer_labels: np.ndarray) -> dict:
    # Build class weights to handle label imbalance.
    unique_classes = np.unique(train_integer_labels)
    class_weights_values = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=train_integer_labels,
    )
    return {int(class_id): float(weight) for class_id, weight in zip(unique_classes, class_weights_values)}


def plot_training_curves(history) -> None:
    # Save training and validation curves.
    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history.get("accuracy", []), label="Train Accuracy")
        plt.plot(history.history.get("val_accuracy", []), label="Validation Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history.get("loss", []), label="Train Loss")
        plt.plot(history.history.get("val_loss", []), label="Validation Loss")
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig(TRAINING_CURVES_PATH, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved training curves to {TRAINING_CURVES_PATH}")
    except Exception as error:
        print(f"Warning: could not save training curves: {error}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray) -> np.ndarray:
    # Generate and save confusion matrix image.
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    try:
        plt.figure(figsize=(18, 14))
        sns.heatmap(
            cm,
            annot=False,
            cmap="viridis",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            linewidths=0.25,
            linecolor="white",
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_PATH, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")
    except Exception as error:
        print(f"Warning: could not save confusion matrix: {error}")
    return cm


def print_per_class_accuracy(cm: np.ndarray, class_names: np.ndarray) -> None:
    # Print per-class accuracy for all classes.
    print("Per-class accuracy:")
    for index, label in enumerate(class_names):
        total = int(cm[index].sum())
        correct = int(cm[index, index])
        accuracy = (correct / total) * 100.0 if total else 0.0
        print(f"  {label}: {accuracy:.2f}% ({correct}/{total})")


def main() -> None:
    # Main training workflow.
    print("Starting improved training pipeline...")

    try:
        ensure_model_directory()
        data_frame = load_dataset(CSV_PATH)
    except Exception as error:
        print(f"Error: {error}")
        return

    # Validate dataset format.
    if data_frame.empty:
        print("Error: dataset is empty.")
        return
    if "label" not in data_frame.columns:
        print("Error: CSV must contain a 'label' column.")
        return

    feature_frame = data_frame.drop(columns=["label"])
    if feature_frame.shape[1] != 63:
        print(f"Error: expected 63 feature columns, found {feature_frame.shape[1]}.")
        return

    # Encode labels and save encoder.
    labels = data_frame["label"].astype(str).values
    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(labels)
    categorical_labels = to_categorical(integer_labels)

    try:
        save_pickle_artifact(ENCODER_PATH, label_encoder, "label encoder")
    except Exception as error:
        print(f"Error: {error}")
        return

    # Split train and test sets.
    features = feature_frame.values.astype(np.float32)
    x_train, x_test, y_train, y_test, train_labels, test_labels, train_integer_labels, test_integer_labels = train_test_split(
        features,
        categorical_labels,
        labels,
        integer_labels,
        test_size=TEST_SPLIT,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    print(f"Training samples: {x_train.shape[0]}")
    print(f"Testing samples: {x_test.shape[0]}")
    print(f"Classes: {len(label_encoder.classes_)}")

    # Fit scaler on train and apply to train/test.
    scaler = StandardScaler()
    try:
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        save_pickle_artifact(SCALER_PATH, scaler, "scaler")
    except Exception as error:
        print(f"Error: scaler processing failed: {error}")
        return

    # Build class weights.
    class_weights = compute_class_weights(train_integer_labels)
    print("Computed class weights for imbalanced classes.")

    # Build and train the model.
    model = build_model(len(label_encoder.classes_))
    model.summary(print_fn=lambda line: print(line))

    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            patience=7,
            factor=0.3,
            min_lr=0.00001,
            verbose=1,
        ),
    ]

    print("Starting model training...")
    history = model.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Reload best checkpoint for final evaluation.
    try:
        print(f"Loading best checkpoint from {MODEL_PATH}...")
        best_model = load_model(MODEL_PATH)
    except Exception as error:
        print(f"Error: unable to load best model: {error}")
        return

    # Evaluate and report metrics.
    print("Evaluating on test split...")
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Final test loss: {test_loss:.4f}")

    probabilities = best_model.predict(x_test, verbose=0)
    predicted_indices = np.argmax(probabilities, axis=1)
    predicted_labels = label_encoder.inverse_transform(predicted_indices)

    print("Classification report:")
    print(classification_report(test_labels, predicted_labels, digits=4, zero_division=0))

    cm = plot_confusion_matrix(test_labels, predicted_labels, label_encoder.classes_)
    print_per_class_accuracy(cm, label_encoder.classes_)
    plot_training_curves(history)

    print(f"Training complete. Best model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
