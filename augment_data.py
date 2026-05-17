import os

import numpy as np
import pandas as pd

# Core dataset path and deterministic seed.
CSV_PATH = "landmarks_data.csv"
RANDOM_STATE = 42


def load_dataset(csv_path: str) -> pd.DataFrame:
    # Load the landmarks CSV with a clear error if the file is missing.
    try:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
        print(f"Loading dataset from {csv_path}...")
        return pd.read_csv(csv_path)
    except Exception as error:
        raise RuntimeError(f"Unable to load dataset: {error}") from error


def validate_dataset(data_frame: pd.DataFrame) -> list:
    # Confirm the dataset shape is suitable for landmark augmentation.
    if data_frame.empty:
        raise ValueError("Landmarks dataset is empty.")
    if "label" not in data_frame.columns:
        raise ValueError("CSV must contain a 'label' column.")
    feature_columns = [column for column in data_frame.columns if column != "label"]
    if len(feature_columns) != 63:
        raise ValueError(f"Expected 63 feature columns, found {len(feature_columns)}.")
    return feature_columns


def apply_noise(landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Add a small amount of Gaussian noise to the full feature vector.
    return landmarks + rng.normal(0.0, 0.005, size=landmarks.shape)


def apply_scaling(landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Apply a mild global scaling factor to simulate slightly different hand sizes.
    scale = rng.uniform(0.95, 1.05)
    return landmarks * scale


def apply_rotation(landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Rotate the x/y coordinates around the hand center by a small angle.
    reshaped = landmarks.reshape(21, 3).copy()
    angle = np.deg2rad(rng.uniform(-10.0, 10.0))
    center_x = float(np.mean(reshaped[:, 0]))
    center_y = float(np.mean(reshaped[:, 1]))

    x_shifted = reshaped[:, 0] - center_x
    y_shifted = reshaped[:, 1] - center_y

    rotated_x = x_shifted * np.cos(angle) - y_shifted * np.sin(angle)
    rotated_y = x_shifted * np.sin(angle) + y_shifted * np.cos(angle)

    reshaped[:, 0] = rotated_x + center_x
    reshaped[:, 1] = rotated_y + center_y
    return reshaped.reshape(-1)


def augment_sample(landmarks: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Combine light transformations so each synthetic row is slightly different.
    augmented = landmarks.astype(np.float32).copy()
    augmented = apply_noise(augmented, rng)
    augmented = apply_scaling(augmented, rng)
    augmented = apply_rotation(augmented, rng)
    augmented = np.clip(augmented, -2.0, 2.0)
    return augmented


def main() -> None:
    # Load and inspect the dataset before adding any synthetic samples.
    try:
        print("Starting augmentation workflow...")
        data_frame = load_dataset(CSV_PATH)
        feature_columns = validate_dataset(data_frame)
    except Exception as error:
        print(f"Error: {error}")
        return

    label_counts = data_frame["label"].value_counts()
    average_count = int(np.ceil(label_counts.mean()))
    weak_labels = label_counts[label_counts < average_count]

    print(f"Found {len(label_counts)} labels in dataset.")
    print(f"Average samples per label: {average_count}")

    if weak_labels.empty:
        print("All labels are already at or above the average sample count. No augmentation needed.")
        return

    rng = np.random.default_rng(RANDOM_STATE)
    augmented_rows = []

    # Augment only the weak labels until each one reaches the average count.
    for label, current_count in weak_labels.items():
        rows_for_label = data_frame[data_frame["label"] == label]
        if rows_for_label.empty:
            continue

        needed = average_count - int(current_count)
        if needed <= 0:
            continue

        source_values = rows_for_label[feature_columns].values.astype(np.float32)
        generated_rows = []
        for _ in range(needed):
            sample_index = int(rng.integers(0, source_values.shape[0]))
            base_sample = source_values[sample_index]
            augmented_sample = augment_sample(base_sample, rng)
            generated_rows.append([label] + [float(value) for value in augmented_sample])

        print(f"Label {label}: had {int(current_count)} samples, added {len(generated_rows)} augmented samples")
        augmented_rows.extend(generated_rows)

    if not augmented_rows:
        print("No weak labels were eligible for augmentation.")
        return

    augmented_frame = pd.DataFrame(augmented_rows, columns=["label"] + feature_columns)
    updated_frame = pd.concat([data_frame, augmented_frame], ignore_index=True)

    try:
        updated_frame.to_csv(CSV_PATH, index=False)
        print(f"Saved augmented dataset back to {CSV_PATH}")
        print(f"New total rows: {len(updated_frame)}")
        print("Augmentation complete.")
    except Exception as error:
        print(f"Error: unable to save augmented dataset: {error}")


if __name__ == "__main__":
    main()
