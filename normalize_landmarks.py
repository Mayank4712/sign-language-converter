import os

import numpy as np
import pandas as pd

# Core normalization file paths.
INPUT_CSV = "landmarks_data.csv"
OUTPUT_CSV = "landmarks_data_normalized.csv"


def normalize_row(values: np.ndarray) -> np.ndarray:
    # Normalize one 63-value row relative to wrist and hand size.
    points = values.reshape(21, 3).astype(np.float32)

    # Step 1: subtract wrist point (landmark 0) from all landmarks.
    wrist = points[0].copy()
    points[:, 0] = points[:, 0] - wrist[0]
    points[:, 1] = points[:, 1] - wrist[1]
    points[:, 2] = points[:, 2] - wrist[2]

    # Step 2: compute max pairwise distance as hand size.
    diff = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    hand_size = float(np.max(distances))

    # Step 3: divide by hand size when valid.
    if hand_size > 1e-8:
        points = points / hand_size

    return points.reshape(-1)


def main() -> None:
    # Load landmarks CSV safely.
    try:
        input_path = os.path.join(INPUT_CSV)
        output_path = os.path.join(OUTPUT_CSV)

        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input CSV not found at {input_path}")

        print(f"Loading landmarks from {input_path}...")
        data_frame = pd.read_csv(input_path)
    except Exception as error:
        print(f"Error: unable to load input CSV: {error}")
        return

    # Validate required schema.
    if data_frame.empty:
        print("Error: input CSV is empty.")
        return
    if "label" not in data_frame.columns:
        print("Error: CSV must contain a 'label' column.")
        return

    feature_columns = [column for column in data_frame.columns if column != "label"]
    if len(feature_columns) != 63:
        print(f"Error: expected 63 feature columns, found {len(feature_columns)}.")
        return

    # Normalize every row and build output DataFrame.
    try:
        normalized_features = []
        feature_values = data_frame[feature_columns].values.astype(np.float32)
        for row in feature_values:
            normalized_features.append(normalize_row(row))

        normalized_frame = pd.DataFrame(normalized_features, columns=feature_columns)
        normalized_frame.insert(0, "label", data_frame["label"].values)
    except Exception as error:
        print(f"Error: unable to normalize landmarks: {error}")
        return

    # Save normalized CSV safely.
    try:
        normalized_frame.to_csv(output_path, index=False)
        print(f"Normalization complete. Saved {len(normalized_frame)} rows.")
    except Exception as error:
        print(f"Error: unable to save normalized CSV: {error}")


if __name__ == "__main__":
    main()